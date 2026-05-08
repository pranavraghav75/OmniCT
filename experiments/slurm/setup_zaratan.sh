#!/usr/bin/env bash
# One-time setup for OmniCT on UMD Zaratan.
# Run this on a Zaratan login node (NOT inside an sbatch job).
#
# What it does:
#   1. Clones the repo into your scratch directory (scratch is the
#      high-IO BeeGFS filesystem; do NOT install / train from $HOME).
#   2. Creates a Python venv on scratch.
#   3. Installs project deps including a CUDA-compatible PyTorch.
#
# Usage:
#   bash setup_zaratan.sh                 # default: clones into ~/scratch.gw/$USER/OmniCT
#   PROJECT_DIR=/path bash setup_zaratan.sh
set -euo pipefail

# --- Knobs you may want to override ---
PROJECT_DIR="${PROJECT_DIR:-${HOME}/scratch.gw/${USER}/OmniCT}"
REPO_URL="${REPO_URL:-https://github.com/pranavraghav75/OmniCT.git}"
PYTHON_MOD="${PYTHON_MOD:-python}"   # `module load python` -> recent Python 3
# --------------------------------------

echo "[setup] Project dir: ${PROJECT_DIR}"

mkdir -p "$(dirname "${PROJECT_DIR}")"
if [[ ! -d "${PROJECT_DIR}/.git" ]]; then
  echo "[setup] Cloning ${REPO_URL} ..."
  git clone "${REPO_URL}" "${PROJECT_DIR}"
fi
cd "${PROJECT_DIR}"

echo "[setup] Loading Python module ..."
module purge || true
module load "${PYTHON_MOD}"
module list

if [[ ! -d .venv ]]; then
  echo "[setup] Creating venv ..."
  python -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

# Isolate from the system spack Python — `module load python` puts
# its site-packages on PYTHONPATH, which leaks an ancient sympy into
# our venv and breaks torch._dynamo.
unset PYTHONPATH
export PYTHONNOUSERSITE=1

echo "[setup] Upgrading pip ..."
pip install --upgrade pip wheel
# Pin a sympy new enough for torch >= 2.5 (needs equal_valued).
pip install --upgrade "sympy>=1.13"

# Install a CUDA build of torch matching Zaratan's CUDA. The cu121 wheels
# work on both A100 and H100 nodes. If you hit a mismatch, swap to
# cu118 or cu124 -- check `nvidia-smi` on a GPU node to see the driver.
echo "[setup] Installing CUDA-enabled torch (cu121) ..."
pip install --index-url https://download.pytorch.org/whl/cu121 \
            "torch>=2.1.0" "torchvision>=0.16.0"

echo "[setup] Installing project dependencies ..."
pip install -r requirements.txt

echo "[setup] Smoke-test (CPU mode, synthetic data) ..."
bash experiments/smoke_test.sh

echo
echo "[setup] DONE."
echo "  Project dir: ${PROJECT_DIR}"
echo "  Activate with: source ${PROJECT_DIR}/.venv/bin/activate"
echo "  Submit a real job:  cd ${PROJECT_DIR} && sbatch experiments/slurm/train.sbatch"
