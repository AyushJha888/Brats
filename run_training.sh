#!/bin/bash
# run_training.sh

# ── W&B auth ──────────────────────────────────────────────────────────────────
export WANDB_API_KEY="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"

# ── activate venv ─────────────────────────────────────────────────────────────
source /workspace/Brats/venv/bin/activate

# ── run training ──────────────────────────────────────────────────────────────
cd /workspace/Brats
nohup python train_script.py > logs/nohup.out 2>&1 &

echo "Training started with PID: $!"
echo "Monitor: tail -f logs/nohup.out"