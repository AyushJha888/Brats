import copy
import json
import logging
import math
import os
import random
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import wandb

from monai.networks.nets import DiffusionModelUNet
from monai.networks.schedulers import DDIMScheduler, DDPMScheduler
from monai.transforms import Compose, EnsureType, RandAffine, RandFlip

# ── paths & seed ─────────────────────────────────────────────────────────────
def _find_data_root(dirname='brain_only'):
    for p in [Path.cwd(), *Path.cwd().parents]:
        candidate = p / dirname
        if candidate.is_dir():
            return candidate
    for p in [Path('/workspace/brats'), Path('/workspace'), Path.home()]:
        candidate = p / dirname
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"Could not find '{dirname}' directory.")

DATA_ROOT = _find_data_root()
LOG_DIR   = Path.cwd() / 'logs'
CKPT_DIR  = LOG_DIR / 'checkpoints'
CACHE_DIR = Path('./slice_cache')
CKPT_DIR.mkdir(parents=True, exist_ok=True)
(LOG_DIR / 'figures').mkdir(parents=True, exist_ok=True)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ── logger ───────────────────────────────────────────────────────────────────
log_path = LOG_DIR / f"train_base_model_{time.strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger("train")
log.info(f"Log file : {log_path}")
log.info(f"Data root: {DATA_ROOT}")
log.info(f"Device   : {DEVICE} | PyTorch: {torch.__version__}")

# ── data helpers ─────────────────────────────────────────────────────────────
def discover_thin_subjects(root, threshold_mm=2.0):
    subjects = []
    for child in sorted(root.iterdir()):
        nii = child / 't1_brain.nii.gz'
        if not (child.is_dir() and nii.exists()):
            continue
        img  = nib.load(str(nii))
        z_sp = float(np.sqrt((img.affine[:3, 2] ** 2).sum()))
        if z_sp < threshold_mm:
            subjects.append(dict(
                id=child.name, path=nii,
                z_sp=round(z_sp, 3),
                shape=tuple(img.header.get_data_shape()[:3]),
            ))
    return subjects


def split_subjects(subjects, train_frac=0.80, val_frac=0.10, seed=42):
    rng    = random.Random(seed)
    names  = sorted(s['id'] for s in subjects)
    rng.shuffle(names)
    lookup  = {s['id']: s for s in subjects}
    ordered = [lookup[n] for n in names]
    n       = len(ordered)
    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)
    return ordered[:n_train], ordered[n_train:n_train+n_val], ordered[n_train+n_val:]


def preprocess_volume(nii_path, target_size=256, z_low=0.15, z_high=0.85, fill_thresh=0.05):
    vol = nib.load(str(nii_path)).get_fdata(dtype=np.float32)
    if vol.ndim != 3:
        return []
    nonzero = vol[vol != 0]
    if nonzero.size == 0:
        return []
    p_lo, p_hi = np.percentile(nonzero, [0.5, 99.5])
    vol = np.clip((vol - p_lo) / max(float(p_hi - p_lo), 1e-6), 0.0, 1.0)
    D   = vol.shape[2]
    z0, z1 = int(np.floor(D * z_low)), int(min(np.ceil(D * z_high), D))
    slices = []
    for zi in range(z0, z1):
        s = vol[:, :, zi]
        if np.count_nonzero(s) / s.size < fill_thresh:
            continue
        if s.shape[0] != target_size or s.shape[1] != target_size:
            t = torch.from_numpy(s[None, None])
            s = F.interpolate(t, (target_size, target_size),
                              mode='bilinear', align_corners=False).squeeze().numpy()
        slices.append(s[np.newaxis].astype(np.float32))
    return slices


def build_disk_cache(subjects, cache_dir=CACHE_DIR, target_size=256):
    cache_dir  = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    index_path = cache_dir / 'slice_index.csv'
    existing   = set(open(index_path).read().splitlines()) if index_path.exists() else set()
    new_rows   = []
    for entry in subjects:
        subj_dir = cache_dir / entry['id']
        if subj_dir.exists() and any(subj_dir.glob('slice_*.npy')):
            continue
        subj_dir.mkdir(exist_ok=True)
        slices = preprocess_volume(entry['path'], target_size=target_size)
        for sl_idx, arr in enumerate(slices):
            fname = subj_dir / f'slice_{sl_idx:04d}.npy'
            row   = str(fname)
            if row not in existing:
                np.save(str(fname), arr)
                new_rows.append(row)
    with open(index_path, 'a') as f:
        for row in new_rows:
            f.write(row + '\n')
    log.info(f"Cache: {len(new_rows)} new slices written" if new_rows else "Cache: all subjects already cached")
    return index_path


def load_index(index_path):
    with open(index_path) as f:
        return [Path(l.strip()) for l in f if l.strip()]


class ThinSliceDataset(Dataset):
    def __init__(self, slice_paths, transform=None):
        self.slice_paths = slice_paths
        self.transform   = transform
        log.info(f"Dataset: {len(self.slice_paths)} slices")

    def __len__(self):
        return len(self.slice_paths)

    def __getitem__(self, idx):
        arr    = np.load(str(self.slice_paths[idx]))
        tensor = torch.from_numpy(arr.copy())
        if self.transform:
            tensor = self.transform(tensor)
        return tensor


# ── build data loaders ────────────────────────────────────────────────────────
all_thin                        = discover_thin_subjects(DATA_ROOT)
train_subjects, val_subjects, _ = split_subjects(all_thin)

train_ids = {s['id'] for s in train_subjects}
val_ids   = {s['id'] for s in val_subjects}
assert len(train_ids & val_ids) == 0, "Subject leakage!"

index_path = CACHE_DIR / 'slice_index.csv'
if index_path.exists():
    log.info("Cache index found, skipping build...")
else:
    log.info("Building disk cache...")
    build_disk_cache(train_subjects + val_subjects)

all_cache_paths = load_index(CACHE_DIR / 'slice_index.csv')
train_paths = [p for p in all_cache_paths if p.parent.name in train_ids]
val_paths   = [p for p in all_cache_paths if p.parent.name in val_ids]

BATCH_SIZE = 8

train_transform = Compose([
    RandFlip(prob=0.5, spatial_axis=1),
    RandAffine(prob=0.5, rotate_range=(np.deg2rad(5),),
               translate_range=(5, 5), padding_mode='zeros', mode='bilinear'),
    EnsureType(dtype=torch.float32),
])
val_transform = Compose([EnsureType(dtype=torch.float32)])

train_loader = DataLoader(ThinSliceDataset(train_paths, train_transform),
                          batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=4, pin_memory=True, drop_last=True)
val_loader   = DataLoader(ThinSliceDataset(val_paths, val_transform),
                          batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=4, pin_memory=True)

log.info(f"Train subjects: {len(train_subjects)} | slices: {len(train_paths)} | batches/epoch: {len(train_loader)}")
log.info(f"Val   subjects: {len(val_subjects)}   | slices: {len(val_paths)}   | batches/epoch: {len(val_loader)}")

# ── model ─────────────────────────────────────────────────────────────────────
unet = DiffusionModelUNet(
    spatial_dims      = 2,
    in_channels       = 1,
    out_channels      = 1,
    channels          = (64, 128, 256, 512),
    attention_levels  = (False, False, False, True),
    num_res_blocks    = 1,
    num_head_channels = 32,
    norm_num_groups   = 32,
).to(DEVICE)

total = sum(p.numel() for p in unet.parameters())
log.info(f"Total params: {total/1e6:.2f} M")

# ── schedulers ────────────────────────────────────────────────────────────────
T = 1000

ddpm_scheduler = DDPMScheduler(
    num_train_timesteps = T,
    schedule            = "linear_beta",
    beta_start          = 1e-4,
    beta_end            = 0.02,
    clip_sample         = False,
)
log.info("Schedulers ready.")

# ── training config ───────────────────────────────────────────────────────────
N_EPOCHS  = 100
LR        = 1e-4
GRAD_CLIP = 1.0

optimizer    = torch.optim.AdamW(unet.parameters(), lr=LR, weight_decay=1e-4)
scheduler_lr = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=N_EPOCHS, eta_min=LR * 0.01
)

log.info(f"N_EPOCHS: {N_EPOCHS} | LR: {LR} | Batch: {BATCH_SIZE}")

# ── W&B init ──────────────────────────────────────────────────────────────────
run = wandb.init(
    entity  = "rahul23082001jha",
    project = "Brats",
    config  = dict(
        n_epochs   = N_EPOCHS,
        lr         = LR,
        grad_clip  = GRAD_CLIP,
        batch_size = BATCH_SIZE,
        seed       = SEED,
        device     = str(DEVICE),
        model      = "DiffusionModelUNet",
        channels   = (64, 128, 256, 512),
        attention   = (False, False, False, True),
        num_res_blocks = 1,
    ),
    settings = wandb.Settings(init_timeout=120),
)
log.info(f"W&B run: {run.url}")

# ── training loop ─────────────────────────────────────────────────────────────
train_losses = []
val_losses   = []

best_val_loss  = float('inf')
best_val_epoch = -1

MAX_CONSECUTIVE_SKIPS = 10
MAX_LOSS              = 2.0
MAX_GRAD_NORM         = 10.0
NAN_EPOCH_LIMIT       = 2

nan_epoch_count = 0

torch.cuda.empty_cache()

for epoch in range(1, N_EPOCHS + 1):
    unet.train()
    epoch_loss        = 0.0
    consecutive_skips = 0
    steps_done        = 0
    t0 = time.perf_counter()

    for batch_idx, batch in enumerate(train_loader):
        t_batch = time.perf_counter()
        x0    = batch.to(DEVICE)
        t     = torch.randint(0, T, (x0.shape[0],), device=DEVICE).long()
        noise = torch.randn_like(x0)
        x_t   = ddpm_scheduler.add_noise(x0, noise, t)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            noise_pred = unet(x_t, t)
            loss       = F.mse_loss(noise_pred, noise)

        if not torch.isfinite(loss):
            optimizer.zero_grad(set_to_none=True)
            consecutive_skips += 1
            if consecutive_skips >= MAX_CONSECUTIVE_SKIPS:
                raise RuntimeError("Training aborted — too many consecutive non-finite losses")
            continue

        if loss.item() > MAX_LOSS:
            raise RuntimeError(f"Training aborted — loss explosion: {loss.item():.4f}")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(unet.parameters(), GRAD_CLIP)

        if not torch.isfinite(grad_norm) or grad_norm.item() > MAX_GRAD_NORM:
            optimizer.zero_grad(set_to_none=True)
            consecutive_skips += 1
            if consecutive_skips >= MAX_CONSECUTIVE_SKIPS:
                raise RuntimeError("Training aborted — too many consecutive bad gradients")
            continue

        optimizer.step()
        epoch_loss        += loss.item()
        consecutive_skips  = 0
        steps_done        += 1

        global_step = (epoch - 1) * len(train_loader) + batch_idx + 1
        if steps_done % 100 == 0 or batch_idx == len(train_loader) - 1:
            log.info(
                f"E{epoch:04d} | step {batch_idx+1:04d}/{len(train_loader)} | "
                f"loss={loss.item():.5f} | grad={grad_norm.item():.3f} | "
                f"t=[{t.min().item()},{t.max().item()}]"
            )
            wandb.log({
                "train/mse_step"  : loss.item(),
                "train/grad_norm" : grad_norm.item(),
                "train/step_time" : time.perf_counter() - t_batch,
            }, step=global_step)

    scheduler_lr.step()

    if steps_done == 0:
        raise RuntimeError("Training aborted — no valid steps in epoch")

    avg_train = epoch_loss / steps_done
    train_losses.append(avg_train)

    if not torch.isfinite(torch.tensor(avg_train)):
        nan_epoch_count += 1
        if nan_epoch_count >= NAN_EPOCH_LIMIT:
            raise RuntimeError("Training aborted — repeated NaN epoch losses")

    # ── val ───────────────────────────────────────────────────────────────────
    unet.eval()
    val_loss  = 0.0
    val_steps = 0
    t_val     = time.perf_counter()

    with torch.no_grad():
        for batch in val_loader:
            x0    = batch.to(DEVICE)
            t     = torch.linspace(0, T-1, x0.shape[0], device=DEVICE).long()
            noise = torch.randn_like(x0)
            x_t   = ddpm_scheduler.add_noise(x0, noise, t)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                noise_pred = unet(x_t, t)
            batch_val = F.mse_loss(noise_pred.float(), noise).item()
            if not torch.isfinite(torch.tensor(batch_val)):
                continue
            val_loss  += batch_val
            val_steps += 1

    avg_val = val_loss / max(val_steps, 1)
    val_losses.append(avg_val)

    mem_alloc  = torch.cuda.memory_allocated(DEVICE) / 1e9
    mem_reserv = torch.cuda.memory_reserved(DEVICE)  / 1e9
    lr_now     = scheduler_lr.get_last_lr()[0]

    log.info(
        f"Epoch {epoch:04d}/{N_EPOCHS} | "
        f"train={avg_train:.5f} | val={avg_val:.5f} | "
        f"best={best_val_loss:.5f} (e{best_val_epoch}) | "
        f"lr={lr_now:.2e} | steps={steps_done}/{len(train_loader)} | "
        f"VRAM={mem_alloc:.1f}/{mem_reserv:.1f}GB | "
        f"time={time.perf_counter()-t0:.1f}s"
    )

    wandb.log({
        "train/mse_epoch"  : avg_train,
        "val/mse"          : avg_val,
        "train/lr"         : lr_now,
        "train/epoch_time" : time.perf_counter() - t0,
        "val/epoch_time"   : time.perf_counter() - t_val,
        "sys/vram_alloc_gb": mem_alloc,
        "sys/vram_resv_gb" : mem_reserv,
    }, step=epoch * len(train_loader))

    # ── best checkpoint ───────────────────────────────────────────────────────
    if avg_val < best_val_loss:
        best_val_loss  = avg_val
        best_val_epoch = epoch
        best_ckpt_path = CKPT_DIR / 'diffusion_best.pt'
        torch.save({
            'epoch'        : epoch,
            'unet'         : unet.state_dict(),
            'optimizer'    : optimizer.state_dict(),
            'train_losses' : train_losses,
            'val_losses'   : val_losses,
        }, best_ckpt_path)
        log.info(f"  ★ New best val={avg_val:.5f} at epoch {epoch}")

        # upload weights-only to W&B (smaller)
        weights_path = CKPT_DIR / f'diffusion_best_e{epoch:04d}_weights.pt'
        torch.save(unet.state_dict(), weights_path)
        art = wandb.Artifact(f"diffusion_best_e{epoch:04d}", type="model")
        art.add_file(str(weights_path))
        wandb.log_artifact(art)
        wandb.log({"val/best_mse": avg_val, "val/best_epoch": epoch},
                  step=epoch * len(train_loader))

    torch.cuda.empty_cache()

wandb.finish()
log.info("Training complete.")
log.info(f"Best val loss: {best_val_loss:.5f} at epoch {best_val_epoch}")