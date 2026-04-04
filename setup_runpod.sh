#!/bin/bash
# setup_venv.sh

set -e

cd /workspace/Brats

# ── create venv ───────────────────────────────────────────────────────────────
echo "Creating virtual environment..."
python3 -m venv .venv

# ── activate ──────────────────────────────────────────────────────────────────
source .venv/bin/activate

# ── upgrade pip ───────────────────────────────────────────────────────────────
pip install --upgrade pip

# ── install requirements ──────────────────────────────────────────────────────
echo "Installing requirements..."
pip install -r requirements.txt

echo "Setup complete."

# ── keep venv active in current shell ─────────────────────────────────────────
exec bash --rcfile <(echo "source /workspace/Brats/.venv/bin/activate")