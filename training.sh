#!/bin/bash
# run_training.sh

# ── W&B auth ─────────────────────────────────────────────────────────────────
export WANDB_API_KEY="your_api_key_here"

# ── activate venv ─────────────────────────────────────────────────────────────
source /workspace/Brats/.venv/bin/activate

# ── run training ──────────────────────────────────────────────────────────────
cd /workspace/Brats
python train.py