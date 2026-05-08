#!/usr/bin/env bash
# Fallback smoke test — same as smoke.sbatch but requests "any GPU"
# instead of a fractional A100. Use this if `smoke.sbatch` errors out
# with "Requested node configuration is not available".

#SBATCH --job-name=omnict-smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=results/slurm-%x-%j.out
#SBATCH --error=results/slurm-%x-%j.err
# #SBATCH --account=your_class_account

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p results

echo "[smoke] host=$(hostname) job=${SLURM_JOB_ID}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

module purge || true
module load python
source .venv/bin/activate

# Isolate from system Python (see smoke.sbatch).
unset PYTHONPATH
export PYTHONNOUSERSITE=1

python -c "import torch, sympy; print('torch', torch.__version__, 'sympy', sympy.__version__, 'cuda?', torch.cuda.is_available(), 'gpus:', torch.cuda.device_count())"

python -m src.training.train \
    --config src/configs/baseline_cnn3d.yaml \
    --override "seed=0" "device=cuda" "train.epochs=1" \
                "data.synthetic_n_train=32" "data.synthetic_n_val=8" \
                "data.spatial_size=[48,48,48]" "train.batch_size=4" "train.num_workers=0" \
    --run_name slurm_smoke

echo "[smoke] done."
