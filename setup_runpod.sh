#!/bin/bash
# ============================================================
#  RunPod Setup Script — BraTS Diffusion Model
#  Run once after pod starts:  bash setup_runpod.sh
#
#  Optional env vars:
#    HF_TOKEN   — HuggingFace token (needed if bucket is private)
#                 export HF_TOKEN=hf_xxxx   before running
#    SKIP_DATA  — set to 1 to skip data download
#                 export SKIP_DATA=1
# ============================================================

set -e   # exit immediately on any error

echo ""
echo "=============================================="
echo "  RunPod Setup — BraTS Diffusion Model"
echo "=============================================="

# ── 0. System info ───────────────────────────────────────────
echo ""
echo "[0/7] System info"
python3 --version
nvcc --version 2>/dev/null | head -1 || echo "  nvcc not found (checking torch for CUDA)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "  nvidia-smi unavailable"
echo ""

# ── 1. Detect CUDA version & pick right torch index ─────────
echo "[1/7] Detecting CUDA version"
CUDA_VER=$(python3 -c "
import subprocess, re
try:
    out = subprocess.check_output(['nvcc','--version']).decode()
    m = re.search(r'release (\d+)\.(\d+)', out)
    if m: print(f'{m.group(1)}{m.group(2)}')
    else: print('121')
except: print('121')
")
echo "  Detected CUDA version tag: cu${CUDA_VER}"

if [ "$CUDA_VER" = "118" ]; then
    TORCH_INDEX="https://download.pytorch.org/whl/cu118"
    TORCH_VER="torch==2.4.1+cu118"
    TV_VER="torchvision==0.19.1+cu118"
    TA_VER="torchaudio==2.4.1+cu118"
else
    # default: CUDA 12.1 (most RunPod images)
    TORCH_INDEX="https://download.pytorch.org/whl/cu121"
    TORCH_VER="torch==2.4.1+cu121"
    TV_VER="torchvision==0.19.1+cu121"
    TA_VER="torchaudio==2.4.1+cu121"
fi

# ── 2. Upgrade pip ───────────────────────────────────────────
echo ""
echo "[2/7] Upgrading pip"
python3 -m pip install --upgrade pip -q

# ── 3. Install PyTorch (CUDA) ────────────────────────────────
echo ""
echo "[3/7] Installing PyTorch with CUDA (${TORCH_VER})"
python3 -m pip install "${TORCH_VER}" "${TV_VER}" "${TA_VER}" \
    --index-url "${TORCH_INDEX}" -q
echo "  Verifying CUDA..."
python3 -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available after install!'
print(f'  torch {torch.__version__}  |  CUDA {torch.version.cuda}  |  GPUs: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f'    GPU {i}: {props.name}  {props.total_memory//1024**2} MB')
"

# ── 4. Install MONAI + generative deps ───────────────────────
echo ""
echo "[4/7] Installing MONAI[all] and utilities"
python3 -m pip install "monai[all]>=1.3" einops lpips -q

# ── 5. Install remaining requirements ────────────────────────
echo ""
echo "[5/8] Installing project requirements"
python3 -m pip install \
    nibabel==5.3.3 \
    numpy \
    scipy \
    matplotlib \
    seaborn \
    tqdm \
    scikit-learn \
    pillow \
    ipywidgets \
    ipykernel \
    "huggingface-hub[hf_xet]" \
    -q

# ── 6. Set up workspace dirs & move notebooks ─────────────────
echo ""
echo "[6/8] Setting up workspace"

WORKDIR="/workspace/brats"
mkdir -p "${WORKDIR}"
cd "${WORKDIR}"

# Move notebooks if uploaded to /workspace root
for nb in 01_data_pipeline.ipynb \
           02_model_setup.ipynb \
           03_slice_thickness_analysis.ipynb \
           04_thin_slice_preprocessing.ipynb \
           05_train_diffusion_lora.ipynb; do
    if [ -f "/workspace/${nb}" ]; then
        mv "/workspace/${nb}" "${WORKDIR}/"
        echo "  Moved ${nb} → ${WORKDIR}"
    fi
done

mkdir -p "${WORKDIR}/logs/figures"
mkdir -p "${WORKDIR}/logs/checkpoints"
mkdir -p "${WORKDIR}/brain_only"

echo "  Workspace: ${WORKDIR}"

# ── 7. Download MR-RATE data from HuggingFace bucket ─────────
echo ""
echo "[7/8] Downloading MR-RATE data from HuggingFace"

HF_BUCKET="hf://buckets/rahul2001/mr-rate-processed"
DATA_DIR="${WORKDIR}/brain_only"

if [ "${SKIP_DATA}" = "1" ]; then
    echo "  SKIP_DATA=1 — skipping download"
else
    # Login with token if provided
    if [ -n "${HF_TOKEN}" ]; then
        echo "  Logging in with HF_TOKEN..."
        python3 -c "from huggingface_hub import login; login('${HF_TOKEN}')"
    else
        echo "  No HF_TOKEN set — attempting anonymous access"
        echo "  (If download fails, run:  export HF_TOKEN=hf_xxx  then re-run)"
    fi

    echo "  Syncing ${HF_BUCKET} → ${DATA_DIR}"
    echo "  Expected: ~421 files, ~3-10 GB depending on modalities"
    echo "  This will take several minutes on a fast connection..."
    echo ""

    python3 - << 'PYEOF'
import os, sys
from huggingface_hub import HfFileSystem

HF_BUCKET  = "hf://buckets/rahul2001/mr-rate-processed"
DATA_DIR   = "/workspace/brats/brain_only"
os.makedirs(DATA_DIR, exist_ok=True)

fs = HfFileSystem()

# list all files in the bucket
try:
    all_files = fs.glob(f"{HF_BUCKET}/**/*.nii.gz")
    print(f"  Found {len(all_files)} files in bucket")
except Exception as e:
    print(f"  ERROR listing bucket: {e}")
    print("  Try setting HF_TOKEN and re-running.")
    sys.exit(1)

downloaded = 0
skipped    = 0
errors     = []

for remote_path in all_files:
    # remote_path looks like:  hf://buckets/rahul2001/mr-rate-processed/SUBJECTID/file.nii.gz
    rel_path   = remote_path.replace(HF_BUCKET.rstrip("/") + "/", "")
    local_path = os.path.join(DATA_DIR, rel_path)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    if os.path.exists(local_path):
        skipped += 1
        continue

    try:
        fs.get(remote_path, local_path)
        downloaded += 1
        if downloaded % 20 == 0:
            print(f"  Progress: {downloaded} downloaded, {skipped} skipped, {len(errors)} errors")
    except Exception as e:
        errors.append((remote_path, str(e)))

print(f"\n  Sync complete:")
print(f"    Downloaded : {downloaded}")
print(f"    Skipped    : {skipped}  (already existed)")
print(f"    Errors     : {len(errors)}")
if errors:
    print("  First 5 errors:")
    for p, err in errors[:5]:
        print(f"    {p}: {err}")

# count subjects
subjects = [d for d in os.listdir(DATA_DIR)
            if os.path.isdir(os.path.join(DATA_DIR, d))
            and os.path.exists(os.path.join(DATA_DIR, d, "t1_brain.nii.gz"))]
print(f"  Subjects with t1_brain.nii.gz : {len(subjects)}")
PYEOF

    echo ""
    echo "  Data directory: ${DATA_DIR}"
fi

# ── 8. Quick smoke test ───────────────────────────────────────
echo ""
echo "[8/8] Smoke test"
python3 - << 'PYEOF'
import torch
import nibabel
import monai
import numpy as np
import matplotlib

# basic check
assert torch.cuda.is_available(), "CUDA not available"
x = torch.randn(1, 1, 256, 256).cuda()

from monai.networks.nets import DiffusionModelUNet
from monai.networks.schedulers import DDIMScheduler, DDPMScheduler

unet = DiffusionModelUNet(
    spatial_dims=2, in_channels=1, out_channels=1,
    num_channels=(128, 256, 256, 512),
    attention_levels=(False, False, True, True),
    num_res_blocks=2, num_head_channels=64,
).cuda()

t  = torch.tensor([100]).cuda()
out = unet(x, t)
assert out.shape == (1, 1, 256, 256), f"Unexpected output shape: {out.shape}"

total = sum(p.numel() for p in unet.parameters())
print(f"  nibabel   {nibabel.__version__}")
print(f"  monai     {monai.__version__}")
print(f"  torch     {torch.__version__}  (CUDA {torch.version.cuda})")
print(f"  UNet fwd  OK  — output {tuple(out.shape)}  params {total/1e6:.1f}M")
print("  ALL CHECKS PASSED")
PYEOF

echo ""
echo "=============================================="
echo "  Setup complete!"
echo "  Workspace  : /workspace/brats"
echo "  Data       : /workspace/brats/brain_only/"
echo ""
echo "  To start Jupyter:"
echo "    jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root"
echo ""
echo "  To re-download data only (skip installs):"
echo "    export SKIP_DATA=0 && bash setup_runpod.sh"
echo ""
echo "  If HF download failed, retry with token:"
echo "    export HF_TOKEN=hf_xxxx && bash setup_runpod.sh"
echo "=============================================="
