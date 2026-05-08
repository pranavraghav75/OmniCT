#!/usr/bin/env bash
# Array job: trains the same config under 3 random seeds in parallel
# (one A100 per seed). Total wall-clock = same as a single run.
#
# Submit:
#   sbatch experiments/slurm/train_array.sbatch                        # default: lora.yaml
#   sbatch --export=CONFIG=src/configs/baseline_cnn3d.yaml \
#          experiments/slurm/train_array.sbatch
#
# To fan out *all four* methods at once, run this script four times
# with --export=CONFIG=... or use experiments/slurm/submit_all.sh.

#SBATCH --job-name=omnict-array
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-2
#SBATCH --output=results/slurm-%x-%A_%a.out
#SBATCH --error=results/slurm-%x-%A_%a.err
# #SBATCH --account=your_class_account

set -euo pipefail

: "${CONFIG:=src/configs/lora.yaml}"
: "${EXTRA_OVERRIDES:=}"
SEED="${SLURM_ARRAY_TASK_ID}"
RUN_NAME="$(basename "${CONFIG}" .yaml)_seed${SEED}"

cd "${SLURM_SUBMIT_DIR}"
mkdir -p results

echo "[array] host=$(hostname) job=${SLURM_JOB_ID} task=${SEED}"
echo "[array] config=${CONFIG} run_name=${RUN_NAME}"

module purge || true
module load python
source .venv/bin/activate
unset PYTHONPATH
export PYTHONNOUSERSITE=1

python -m src.training.train \
    --config "${CONFIG}" \
    --override "seed=${SEED}" "device=cuda" ${EXTRA_OVERRIDES} \
    --run_name "${RUN_NAME}"

echo "[array] done."
cat "results/${RUN_NAME}/metrics.json"
