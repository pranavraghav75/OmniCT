#!/usr/bin/env bash
# Cheap GPU smoke test on Zaratan.
#
# We use the `gpu` partition (the `debug` partition is CPU-only) with
# the *cheapest* GPU SLURM offers: a fractional 1/7 A100 (5 GB GPU
# RAM, 7 SU/hr). If your allocation doesn't have access to fractional
# A100s, the fallback is `--gres=gpu:1` (any GPU). The job is small —
# 10 min walltime, ~2 GB GPU RAM in practice — so a fractional A100 is
# more than enough.
#
# Submit:
#   sbatch experiments/slurm/smoke.sbatch
# Or, if fractional GPUs aren't available to your account:
#   sbatch --gres=gpu:1 experiments/slurm/smoke.sbatch

#SBATCH --job-name=omnict-smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100_1g.5gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=results/slurm-%x-%j.out
#SBATCH --error=results/slurm-%x-%j.err
# Uncomment + fill in if your account is not the default:
# #SBATCH --account=your_class_account

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p results

echo "[smoke] host=$(hostname) job=${SLURM_JOB_ID}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

module purge || true
module load python
source .venv/bin/activate

# `module load python` injects the system spack site-packages into
# PYTHONPATH, which leaks ancient sympy / numpy into our venv and
# breaks torch. Strip everything system-Python so only the venv is
# searched.
unset PYTHONPATH
export PYTHONNOUSERSITE=1

python -c "import torch, sympy; print('torch', torch.__version__, 'sympy', sympy.__version__, 'cuda?', torch.cuda.is_available(), 'gpus:', torch.cuda.device_count())"

# Tiny synthetic-data run on a real GPU; verifies CUDA + the whole
# training loop end-to-end. Should finish in <1 min.
python -m src.training.train \
    --config src/configs/baseline_cnn3d.yaml \
    --override "seed=0" "device=cuda" "train.epochs=1" \
                "data.synthetic_n_train=32" "data.synthetic_n_val=8" \
                "data.spatial_size=[48,48,48]" "train.batch_size=4" "train.num_workers=0" \
    --run_name slurm_smoke

echo "[smoke] done."
