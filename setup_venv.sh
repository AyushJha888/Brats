#!/bin/bash
# setup_venv.sh

set -e

cd /workspace/Brats

# ── install uv if not present ─────────────────────────────────────────────────
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env
fi

# ── create venv ───────────────────────────────────────────────────────────────
echo "Creating virtual environment..."
uv venv .venv

# ── activate ──────────────────────────────────────────────────────────────────
source .venv/bin/activate

# ── install torch first (special index) ───────────────────────────────────────
echo "Installing PyTorch for CUDA 12.8 (Blackwell)..."
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# ── install remaining requirements ────────────────────────────────────────────
echo "Installing requirements..."
uv pip install -r requirements.txt

echo "Setup complete."

exec bash --rcfile <(echo "source /workspace/Brats/.venv/bin/activate")#!/bin/bash
# setup_venv.sh

set -e

cd /workspace/Brats

# ── install uv if not present ─────────────────────────────────────────────────
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env
fi

# ── create venv ───────────────────────────────────────────────────────────────
echo "Creating virtual environment..."
uv venv .venv

# ── activate ──────────────────────────────────────────────────────────────────
source .venv/bin/activate

# ── install torch first (special index) ───────────────────────────────────────
echo "Installing PyTorch for CUDA 12.8 (Blackwell)..."
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# ── install remaining requirements ────────────────────────────────────────────
echo "Installing requirements..."
uv pip install -r requirements.txt

echo "Setup complete."

exec bash --rcfile <(echo "source /workspace/Brats/.venv/bin/activate")