#!/usr/bin/env bash
# Generic OmniCT training job for UMD Zaratan.
#
# Submit:
#   sbatch experiments/slurm/train.sbatch                            # default: lora seed=0
#   sbatch --export=CONFIG=src/configs/baseline_cnn3d.yaml,SEED=1 \
#          experiments/slurm/train.sbatch
#
# Or to fan out over 3 seeds with one command, use the array job below.

#SBATCH --job-name=omnict-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=results/slurm-%x-%A_%a.out
#SBATCH --error=results/slurm-%x-%A_%a.err
# Uncomment + fill in if your account is non-default:
# #SBATCH --account=your_class_account

set -euo pipefail

# --- Job knobs (overridable via --export=KEY=VALUE) ---
: "${CONFIG:=src/configs/lora.yaml}"
: "${SEED:=0}"
: "${RUN_NAME:=$(basename "${CONFIG}" .yaml)_seed${SEED}}"
: "${EXTRA_OVERRIDES:=}"                # e.g. "data.synthetic=false train.epochs=20"
# ------------------------------------------------------

cd "${SLURM_SUBMIT_DIR}"
mkdir -p results

echo "[train] host=$(hostname) job=${SLURM_JOB_ID} array=${SLURM_ARRAY_TASK_ID:-N/A}"
echo "[train] config=${CONFIG} seed=${SEED} run_name=${RUN_NAME}"
echo "[train] gpu="; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

module purge || true
module load python
# Some Zaratan torch wheels need the bundled cuda runtime; usually no
# extra cuda module is needed if you `pip install`'d a cu121 build.
# module load cuda     # uncomment if torch can't find libcuda

source .venv/bin/activate

# `module load python` injects the system spack site-packages into
# PYTHONPATH, which leaks ancient sympy / numpy into our venv and
# breaks torch. Strip everything system-Python so only the venv is
# searched.
unset PYTHONPATH
export PYTHONNOUSERSITE=1

python -m src.training.train \
    --config "${CONFIG}" \
    --override "seed=${SEED}" "device=cuda" ${EXTRA_OVERRIDES} \
    --run_name "${RUN_NAME}"

echo "[train] done. metrics:"
cat "results/${RUN_NAME}/metrics.json"
