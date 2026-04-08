"""
train_simplex.py
================
Diffusion model training with fractal/simplex noise.
Converted from 09_train_simplex_noise.ipynb.

Usage:
    python train_simplex.py
    python train_simplex.py --epochs 300 --batch_size 16 --lr 1e-4
    python train_simplex.py --resume /path/to/checkpoint.pt
"""

import argparse
import copy
import json
import logging
import math
import random
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir',    type=str,   default='brain_only')
    p.add_argument('--cache_dir',   type=str,   default='./slice_cache')
    p.add_argument('--log_dir',     type=str,   default='./logs')
    p.add_argument('--epochs',      type=int,   default=300)
    p.add_argument('--batch_size',  type=int,   default=8)
    p.add_argument('--lr',          type=float, default=1e-4)
    p.add_argument('--grad_clip',   type=float, default=1.0)
    p.add_argument('--save_every',  type=int,   default=5)
    p.add_argument('--seed',        type=int,   default=42)
    p.add_argument('--octaves',     type=int,   default=6)
    p.add_argument('--resume',      type=str,   default=None,
                   help='Path to checkpoint to resume from')
    p.add_argument('--wandb_entity',type=str,   default='rahul23082001jha')
    p.add_argument('--wandb_project',type=str,  default='Brats')
    p.add_argument('--no_wandb',    action='store_true')
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
def find_data_root(dirname):
    for p in [Path.cwd(), *Path.cwd().parents]:
        c = p / dirname
        if c.is_dir():
            return c
    for p in [Path('/workspace/brats'), Path('/workspace'), Path.home()]:
        c = p / dirname
        if c.is_dir():
            return c
    raise FileNotFoundError(f"Could not find '{dirname}' directory.")


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────
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
    z0  = int(np.floor(D * z_low))
    z1  = int(min(np.ceil(D * z_high), D))
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


def build_disk_cache(subjects, cache_dir, target_size=256):
    cache_dir  = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    index_path = cache_dir / 'slice_index.csv'

    existing = set(open(index_path).read().splitlines()) if index_path.exists() else set()
    new_rows = []

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

    if new_rows:
        print(f"Cache: {len(new_rows)} new slices written to {cache_dir}")
    else:
        print("Cache: all subjects already cached, nothing written")
    return index_path


def load_index(index_path):
    with open(index_path) as f:
        return [Path(l.strip()) for l in f if l.strip()]


class ThinSliceDataset(Dataset):
    def __init__(self, slice_paths, transform=None):
        self.slice_paths = slice_paths
        self.transform   = transform

    def __len__(self):
        return len(self.slice_paths)

    def __getitem__(self, idx):
        arr    = np.load(str(self.slice_paths[idx]))
        tensor = torch.from_numpy(arr.copy())
        if self.transform:
            tensor = self.transform(tensor)
        return tensor


# ─────────────────────────────────────────────────────────────────────────────
# Simplex / fractal noise
# ─────────────────────────────────────────────────────────────────────────────
def generate_simplex_noise(shape, octaves=6, device='cpu'):
    """
    Fractal noise: sum of Gaussian noise at multiple spatial frequencies.
    Normalised per-sample to N(0,1) so DDPM SNR math stays correct.
    Reference: Wolleb et al. 'Diffusion Models for Medical Anomaly Detection' (2022)
    """
    B, C, H, W = shape
    noise = torch.zeros(B, C, H, W, device=device)
    for octave in range(octaves):
        freq      = 2 ** octave
        amplitude = 0.5 ** octave
        sh = max(1, H // freq)
        sw = max(1, W // freq)
        raw = torch.randn(B, C, sh, sw, device=device)
        if sh != H or sw != W:
            raw = F.interpolate(raw, (H, W), mode='bilinear', align_corners=False)
        noise = noise + amplitude * raw
    noise = noise - noise.mean(dim=(-1, -2), keepdim=True)
    noise = noise / (noise.std(dim=(-1, -2), keepdim=True) + 1e-8)
    return noise


# ─────────────────────────────────────────────────────────────────────────────
# Model & schedulers
# ─────────────────────────────────────────────────────────────────────────────
def build_model(device):
    unet = DiffusionModelUNet(
    spatial_dims      = 2,
    in_channels       = 1,
    out_channels      = 1,
    channels          = (64, 128, 256, 512),
    attention_levels  = (False, False, False, True),
    num_res_blocks    = 1,
    num_head_channels = 32,
    norm_num_groups   = 32,
    ).to(device)
    total = sum(p.numel() for p in unet.parameters())
    print(f'Model: {total/1e6:.2f} M parameters')
    return unet


def build_schedulers(T=1000):
    kwargs = dict(
        num_train_timesteps = T,
        schedule            = 'linear_beta',
        beta_start          = 1e-4,
        beta_end            = 0.02,
        clip_sample         = False,
    )
    return DDPMScheduler(**kwargs), DDIMScheduler(**kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────
def train(args):
    # ── reproducibility ───────────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    T      = 1000

    # ── dirs ──────────────────────────────────────────────────────────────────
    log_dir  = Path(args.log_dir)
    ckpt_dir = log_dir / 'checkpoints'
    fig_dir  = log_dir / 'figures'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ── logging ───────────────────────────────────────────────────────────────
    log_path = log_dir / f"train_simplex_{time.strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level   = logging.INFO,
        format  = '%(asctime)s | %(levelname)s | %(message)s',
        datefmt = '%H:%M:%S',
        handlers= [
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout),
        ]
    )
    log = logging.getLogger('train')
    log.info(f'Log: {log_path}')
    log.info(f'Device: {DEVICE} | T: {T} | Epochs: {args.epochs} | LR: {args.lr}')
    log.info(f'Batch size: {args.batch_size} | Grad clip: {args.grad_clip} | Seed: {args.seed}')
    # ── data ──────────────────────────────────────────────────────────────────
    data_root = find_data_root(args.data_dir)
    log.info(f'DATA_ROOT: {data_root}')

    all_subjects                     = discover_thin_subjects(data_root)
    train_subjects, val_subjects, _  = split_subjects(all_subjects, seed=args.seed)

    train_ids = {s['id'] for s in train_subjects}
    val_ids   = {s['id'] for s in val_subjects}
    assert len(train_ids & val_ids) == 0, 'Subject leakage!'

    cache_dir  = Path(args.cache_dir)
    index_path = cache_dir / 'slice_index.csv'
    if index_path.exists():
        log.info('Cache index found, skipping build...')
    else:
        log.info('Building disk cache...')
        build_disk_cache(train_subjects + val_subjects, cache_dir)

    all_paths   = load_index(index_path)
    train_paths = [p for p in all_paths if p.parent.name in train_ids]
    val_paths   = [p for p in all_paths if p.parent.name in val_ids]

    train_transform = Compose([
        RandFlip(prob=0.5, spatial_axis=1),
        RandAffine(prob=0.5, rotate_range=(np.deg2rad(5),),
                   translate_range=(5, 5), padding_mode='zeros', mode='bilinear'),
        EnsureType(dtype=torch.float32),
    ])
    val_transform = Compose([EnsureType(dtype=torch.float32)])

    train_loader = DataLoader(
        ThinSliceDataset(train_paths, train_transform),
        batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        ThinSliceDataset(val_paths, val_transform),
        batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    log.info(f'Train: {len(train_subjects)} subjects | {len(train_paths)} slices | '
             f'{len(train_loader)} batches/epoch')
    log.info(f'Val  : {len(val_subjects)} subjects | {len(val_paths)} slices | '
             f'{len(val_loader)} batches/epoch')

    # ── model & optimiser ─────────────────────────────────────────────────────
    unet = build_model(DEVICE)
    ddpm_scheduler, ddim_scheduler = build_schedulers(T)

    optimizer    = torch.optim.AdamW(unet.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler_lr = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    # ── resume ────────────────────────────────────────────────────────────────
    start_epoch  = 1
    train_losses = []
    val_losses   = []
    best_val_loss  = float('inf')
    best_val_epoch = -1

    if args.resume:
        log.info(f'Resuming from {args.resume}')
        ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
        unet.load_state_dict(ckpt['unet'])
        optimizer.load_state_dict(ckpt['optimizer'])
        train_losses   = ckpt.get('train_losses', [])
        val_losses     = ckpt.get('val_losses', [])
        start_epoch    = ckpt.get('epoch', 0) + 1
        best_val_loss  = min(val_losses) if val_losses else float('inf')
        best_val_epoch = val_losses.index(best_val_loss) + 1 if val_losses else -1
        log.info(f'Resumed from epoch {start_epoch - 1}  |  best val={best_val_loss:.5f}')

    unet = unet.to(torch.bfloat16)

    # ── W&B ───────────────────────────────────────────────────────────────────
    use_wandb = not args.no_wandb
    if use_wandb:
        run = wandb.init(
            entity  = args.wandb_entity,
            project = args.wandb_project,
            config  = dict(
                epochs          = args.epochs,
                lr              = args.lr,
                grad_clip       = args.grad_clip,
                batch_size      = args.batch_size,
                seed            = args.seed,
                device          = DEVICE,
                model           = 'DiffusionModelUNet',
                channels        = (64, 128, 256, 512),
                noise_type      = 'simplex',
                simplex_octaves = args.octaves,
                resume          = args.resume,
            ),
            resume = 'allow',
        )
        log.info(f'W&B run: {run.url}')

    # ── training loop ─────────────────────────────────────────────────────────
    MAX_CONSECUTIVE_SKIPS = 10
    MAX_LOSS              = 2.0
    MAX_GRAD_NORM         = 10.0
    NAN_EPOCH_LIMIT       = 2
    nan_epoch_count       = 0

    torch.cuda.empty_cache()

    for epoch in range(start_epoch, args.epochs + 1):
        unet.train()
        epoch_loss        = 0.0
        consecutive_skips = 0
        steps_done        = 0
        t0 = time.perf_counter()

        for batch_idx, batch in enumerate(train_loader):
            t_batch = time.perf_counter()
            x0    = batch.to(DEVICE)
            t     = torch.randint(0, T, (x0.shape[0],), device=DEVICE).long()
            noise = generate_simplex_noise(x0.shape, octaves=args.octaves, device=DEVICE)
            x_t   = ddpm_scheduler.add_noise(x0, noise, t)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                noise_pred = unet(x_t, t)
                loss       = F.mse_loss(noise_pred, noise)

            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                consecutive_skips += 1
                if consecutive_skips >= MAX_CONSECUTIVE_SKIPS:
                    raise RuntimeError('Training aborted — too many consecutive non-finite losses')
                continue

            if loss.item() > MAX_LOSS:
                raise RuntimeError(f'Training aborted — loss explosion: {loss.item():.4f}')

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(unet.parameters(), args.grad_clip)

            if not torch.isfinite(grad_norm) or grad_norm.item() > MAX_GRAD_NORM:
                optimizer.zero_grad(set_to_none=True)
                consecutive_skips += 1
                if consecutive_skips >= MAX_CONSECUTIVE_SKIPS:
                    raise RuntimeError('Training aborted — too many consecutive bad gradients')
                continue

            optimizer.step()
            epoch_loss        += loss.item()
            consecutive_skips  = 0
            steps_done        += 1

            global_step = (epoch - 1) * len(train_loader) + batch_idx + 1
            if steps_done % 100 == 0 or batch_idx == len(train_loader) - 1:
                log.info(
                    f'E{epoch:04d} | step {batch_idx+1:04d}/{len(train_loader)} | '
                    f'loss={loss.item():.5f} | grad={grad_norm.item():.3f} | '
                    f't=[{t.min().item()},{t.max().item()}]'
                )
                if use_wandb:
                    wandb.log({
                        'train/mse_step'  : loss.item(),
                        'train/grad_norm' : grad_norm.item(),
                        'train/step_time' : time.perf_counter() - t_batch,
                    }, step=global_step)

        scheduler_lr.step()

        if steps_done == 0:
            raise RuntimeError('Training aborted — no valid steps in epoch')

        avg_train = epoch_loss / steps_done
        train_losses.append(avg_train)

        if not torch.isfinite(torch.tensor(avg_train)):
            nan_epoch_count += 1
            if nan_epoch_count >= NAN_EPOCH_LIMIT:
                raise RuntimeError('Training aborted — repeated NaN epoch losses')

        # ── validation ────────────────────────────────────────────────────────
        unet.eval()
        val_loss  = 0.0
        val_steps = 0
        t_val     = time.perf_counter()

        with torch.no_grad():
            for batch in val_loader:
                x0    = batch.to(DEVICE)
                t     = torch.linspace(0, T - 1, x0.shape[0], device=DEVICE).long()
                noise = generate_simplex_noise(x0.shape, octaves=args.octaves, device=DEVICE)
                x_t   = ddpm_scheduler.add_noise(x0, noise, t)
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    noise_pred = unet(x_t, t)
                bv = F.mse_loss(noise_pred.float(), noise).item()
                if not torch.isfinite(torch.tensor(bv)):
                    continue
                val_loss  += bv
                val_steps += 1

        avg_val = val_loss / max(val_steps, 1)
        val_losses.append(avg_val)

        mem_alloc  = torch.cuda.memory_allocated(DEVICE) / 1e9
        mem_reserv = torch.cuda.memory_reserved(DEVICE)  / 1e9
        lr_now     = scheduler_lr.get_last_lr()[0]

        log.info(
            f'Epoch {epoch:04d}/{args.epochs} | '
            f'train={avg_train:.5f} | val={avg_val:.5f} | '
            f'best={best_val_loss:.5f} (e{best_val_epoch}) | '
            f'lr={lr_now:.2e} | steps={steps_done}/{len(train_loader)} | '
            f'VRAM={mem_alloc:.1f}/{mem_reserv:.1f}GB | '
            f'time={time.perf_counter()-t0:.1f}s'
        )

        if use_wandb:
            wandb.log({
                'train/mse_epoch'  : avg_train,
                'val/mse'          : avg_val,
                'train/lr'         : lr_now,
                'train/epoch_time' : time.perf_counter() - t0,
                'val/epoch_time'   : time.perf_counter() - t_val,
                'sys/vram_alloc_gb': mem_alloc,
                'sys/vram_resv_gb' : mem_reserv,
            }, step=epoch * len(train_loader))

        # ── best checkpoint ───────────────────────────────────────────────────
        if avg_val < best_val_loss:
            best_val_loss  = avg_val
            best_val_epoch = epoch
            best_ckpt_path = ckpt_dir / f'simplex_best_e{epoch:04d}.pt'
            torch.save({
                'epoch'          : epoch,
                'unet'           : unet.state_dict(),
                'optimizer'      : optimizer.state_dict(),
                'train_losses'   : train_losses,
                'val_losses'     : val_losses,
                'noise_type'     : 'simplex',
                'simplex_octaves': args.octaves,
            }, best_ckpt_path)
            log.info(f'  ★ New best val={avg_val:.5f} at epoch {epoch} → {best_ckpt_path.name}')

            if use_wandb:
                wandb.log({'val/best_mse': avg_val, 'val/best_epoch': epoch},
                          step=epoch * len(train_loader))
                art = wandb.Artifact('simplex_diffusion_best', type='model')
                art.add_file(str(best_ckpt_path))
                wandb.log_artifact(art)

        # ── periodic checkpoint ───────────────────────────────────────────────
        if epoch % args.save_every == 0:
            periodic_path = ckpt_dir / f'simplex_epoch{epoch:04d}.pt'
            torch.save({
                'epoch'          : epoch,
                'unet'           : unet.state_dict(),
                'optimizer'      : optimizer.state_dict(),
                'train_losses'   : train_losses,
                'val_losses'     : val_losses,
                'noise_type'     : 'simplex',
                'simplex_octaves': args.octaves,
            }, periodic_path)
            log.info(f'  Periodic checkpoint → {periodic_path.name}')

        torch.cuda.empty_cache()

    # ── final artefacts ───────────────────────────────────────────────────────
    # loss curve
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(train_losses, label='Train MSE', color='#5b9bd5')
    ax.plot(val_losses,   label='Val MSE',   color='#e07b54')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE loss')
    ax.set_title('Simplex noise diffusion — training loss')
    ax.legend()
    plt.tight_layout()
    curve_path = fig_dir / 'simplex_loss_curve.png'
    fig.savefig(curve_path, dpi=150)
    plt.close(fig)
    log.info(f'Loss curve saved → {curve_path}')

    # JSON log
    json_path = log_dir / 'simplex_training_log.json'
    with open(json_path, 'w') as f:
        json.dump({'train_losses': train_losses, 'val_losses': val_losses}, f, indent=2)
    log.info(f'Loss log saved → {json_path}')

    if use_wandb:
        wandb.finish()

    log.info('Training complete.')
    log.info(f'Best val={best_val_loss:.5f} at epoch {best_val_epoch}')


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    args = parse_args()
    train(args)