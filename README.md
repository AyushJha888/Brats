# Brain MRI Anomaly Detection via Diffusion Models

Unsupervised anomaly detection on brain MRI using a **2D DDPM diffusion model with LoRA fine-tuning**, trained exclusively on healthy T1-weighted scans. Pathology is exposed as the residual between a diseased input slice and its healthy reconstruction.

---

## How It Works

```
Diseased slice  x₀
      │
      ▼
Saliency map from ACAT classifier  →  binary mask M
      │  (rough map of "where pathology probably is")
      ▼
DDIM encode  →  x_L
      (forward noising applied only inside mask M, to noise level L)
      │
      ▼
Hybrid DDPM + DDIM decode  (t = L → 0)
      ├─ inside  M = 1  →  DDPM steps  (stochastic, free to regenerate healthy anatomy)
      └─ outside M = 0  →  DDIM steps  (deterministic, preserves healthy anatomy exactly)
      │
      ▼
Healthy reconstruction  x̂₀
      │
      ▼
Anomaly map  =  | x₀ − x̂₀ |
```

The model is trained only on healthy brains. When asked to reconstruct a diseased region (via DDPM decode inside the pathology mask), it generates the most likely *healthy* anatomy for that location. The pixel-wise residual is the anomaly signal.

---

## Project Structure

```
Brats/
├── brain_only/                        # MR-RATE healthy subjects (downloaded)
│   └── <SUBJECT_ID>/
│       ├── t1_brain.nii.gz
│       ├── t2_axi_brain.nii.gz        # (optional)
│       ├── flair_brain.nii.gz         # (optional)
│       └── brain_mask.nii.gz          # (optional)
│
├── BraTS2020_TrainingData/            # BraTS glioma cases (evaluation only)
│   └── MICCAI_BraTS2020_TrainingData/
│       └── BraTS20_Training_XXX/
│           ├── *_flair.nii
│           ├── *_t1.nii
│           ├── *_t1ce.nii
│           ├── *_t2.nii
│           └── *_seg.nii
│
├── logs/
│   ├── figures/                       # plots saved by notebooks
│   ├── checkpoints/                   # model checkpoints from training
│   ├── data_validation.json
│   ├── thin_slice_split.json          # subject IDs per split
│   └── lora_training_log.json         # train/val loss per epoch
│
├── 01_data_pipeline.ipynb             # data discovery, preprocessing checks, DataLoader
├── 02_model_setup.ipynb               # model architecture, forward/backward pass verification
├── 03_slice_thickness_analysis.ipynb  # thick vs thin acquisition analysis, fill-rate stats
├── 04_thin_slice_preprocessing.ipynb  # canonical preprocess_volume fn + ThinSliceDataset
├── 05_train_diffusion_lora.ipynb      # LoRA training loop + hybrid inference pipeline
│
├── requirements.txt
├── setup_runpod.sh                    # one-shot RunPod setup + data download
└── README.md
```

---

## Quickstart

### Local (Windows / Mac)

```bash
# 1. create a Python 3.9 environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. install dependencies
pip install -r requirements.txt

# 3. launch Jupyter
jupyter lab
```

Run notebooks in order: `01` → `02` → `03` → `04` → `05`.

### RunPod (GPU)

```bash
# upload setup_runpod.sh to your pod, then:
bash setup_runpod.sh

# with a HuggingFace token (if needed):
export HF_TOKEN=hf_xxxx
bash setup_runpod.sh

# skip data download (re-run installs only):
export SKIP_DATA=1
bash setup_runpod.sh

# start Jupyter after setup:
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

The script handles CUDA version detection, PyTorch install, MONAI, all dependencies, workspace creation, and data download automatically.

---

## Dataset

### MR-RATE (training) — `brain_only/`

| Property | Value |
|----------|-------|
| Source | `hf://buckets/rahul2001/mr-rate-processed` |
| Total subjects | 212 |
| Modalities | T1w (all), T2/FLAIR (subset) |
| Format | NIfTI `.nii.gz`, skull-stripped |

#### Acquisition types discovered

| Type | Subjects | Z-spacing | Depth | Used? |
|------|----------|-----------|-------|-------|
| Thick-slice (clinical) | 139 | 4–7 mm | 20–30 | ❌ Excluded |
| **Thin-slice (isotropic)** | **71** | **0.47–1.04 mm** | **200–640** | **✅ Training** |

Thick-slice volumes are excluded because partial-volume effects make their slices look fundamentally different from thin-slice. Mixing both types would corrupt the model's notion of normal brain appearance.

#### Thin-slice split (seed=42)

| Split | Subjects | Usable 2D slices |
|-------|----------|-----------------|
| Train | 55 | 9,540 |
| Val | 7 | 1,125 |
| Test | 9 | 1,995 |
| **Total** | **71** | **12,660** |

### BraTS 2020 (evaluation) — `BraTS2020_TrainingData/`

371 glioma cases with `flair / t1 / t1ce / t2 / seg` volumes. Used for evaluating anomaly maps against ground-truth segmentations — **not in the training pipeline**.

---

## Preprocessing

**Canonical function:** `preprocess_volume(nii_path)` → `List[Tensor(1, 256, 256)]`

Defined in `04_thin_slice_preprocessing.ipynb`, reused in `05_train_diffusion_lora.ipynb`.

| Step | Operation |
|------|-----------|
| 1 | Load NIfTI → float32 array (H, W, D) |
| 2 | Percentile normalisation on non-zero voxels: clip [p0.5, p99.5] → [0, 1] |
| 3 | Z-range filter: keep slices in [15%, 85%] of total depth |
| 4 | Fill filter: discard slices with < 5% non-zero pixels |
| 5 | Bilinear resize to 256×256 if shape differs |
| 6 | Return as list of `(1, 256, 256)` float32 tensors |

#### Slice retention funnel

```
All 210 subjects — raw axial slices    :  28,412  (100%)
  After 15%–85% Z-range filter         :  20,074  (−29.3%)
  After 5% brain-fill filter           :  15,119  (−24.7% of candidates)

Thin-slice only (71 subjects):
  Candidate slices in 15%–85% window   :  17,594
  After 5% fill filter                 :  12,660  (72.0% retained)
```

#### Augmentations (train split only)

| Transform | Config |
|-----------|--------|
| `RandFlip` | horizontal, p=0.5 |
| `RandAffine` | rotate ±5°, translate ±5px, p=0.5 |
| `ScaleIntensity` | re-clamps to [0, 1] after affine |

---

## Model

### Architecture

| Component | Value |
|-----------|-------|
| Framework | MONAI `DiffusionModelUNet` (2D) |
| Input / output | 1 channel (T1w grayscale), 256×256 |
| Channels | 128 → 256 → 256 → 512 |
| Attention | Levels 3 & 4 only |
| Residual blocks | 2 per level |
| Head channels | 64 |
| Total parameters | ~102 M |
| Loss | MSE(ε̂, ε) — noise prediction |

### Schedulers

| Scheduler | Used for |
|-----------|----------|
| `DDPMScheduler` | Training + inside-mask decode |
| `DDIMScheduler` | DDIM encode + outside-mask decode |
| Both | Linear β, β_start=1e-4, β_end=0.02, T=1000 |

---

## LoRA Fine-Tuning

Full fine-tuning of 102 M parameters is expensive. LoRA injects low-rank side branches into attention linear layers only:

```
W' x  =  W x  +  B(Ax) · (α / r)
```

Only `A` and `B` are trained. `W` is frozen.

| Parameter | Value |
|-----------|-------|
| Rank `r` | 4 |
| Alpha `α` | 4.0 (scale = 1.0) |
| Target layers | All `nn.Linear` in attention blocks |
| Trainable params | ~0.8 M (<1% of 102 M) |
| Init | A: kaiming_uniform · B: zeros |
| Optimizer | AdamW, lr=1e-4 |
| LR schedule | Cosine annealing |
| Grad clip | 1.0 |

Checkpoints are saved in two forms:
- **Full** — entire model state, used to resume training
- **LoRA-only** — just the A+B matrices (~few MB), loaded on top of a fresh base model for inference

---

## Inference

Three functions implement the full pipeline (`05_train_diffusion_lora.ipynb`):

| Function | Purpose |
|----------|---------|
| `ddim_encode(x0, mask, scheduler, encode_timestep=500)` | Forward-noise x₀ → x_L inside mask only |
| `hybrid_decode(x_L, mask, unet, ddpm, ddim, start_t=500, n_steps=50)` | DDPM inside mask · DDIM outside mask |
| `anomaly_map(x0, x_hat, smooth_sigma=2.0)` | `\|x₀ − x̂₀\|` with optional Gaussian smoothing |

**Key tuning knobs:**

| Parameter | Effect |
|-----------|--------|
| `ENCODE_T` ∈ {250, 500, 750} | Higher = more freedom to replace pathology |
| `N_DDIM_STEPS` | 50 is standard; lower is faster, noisier |
| `smooth_sigma` | Smoothing on the anomaly map (pixels) |

---

## Requirements

**Python 3.9.x** (tested on 3.9.13).

```bash
pip install -r requirements.txt
```

Key packages: `torch==2.4.1`, `monai[all]>=1.3`, `nibabel==5.3.3`, `einops`, `lpips`.

#### GPU variants

```bash
# NVIDIA CUDA 12.1
pip install torch==2.4.1+cu121 torchvision==0.19.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# NVIDIA CUDA 11.8
pip install torch==2.4.1+cu118 torchvision==0.19.1+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# Windows DirectML (AMD / Intel / any DX12 GPU)
pip install torch-directml==0.2.5.dev240914
```

---

## Next Steps

| Priority | Task |
|----------|------|
| 1 | Train `05_train_diffusion_lora.ipynb` on GPU — target 500–1000 epochs |
| 2 | Integrate ACAT classifier to generate real pathology masks |
| 3 | Evaluate anomaly maps against BraTS segmentations (Dice, AUC-ROC) |
| 4 | Tune `ENCODE_T` ∈ {250, 500, 750} and LoRA rank ∈ {4, 8, 16} |
| 5 | Extend to multi-modal input (T2, FLAIR) |
