#!/usr/bin/env bash
# Generate the saliency / 3D Grad-CAM panel (Figure 3 of the report).
# Cheap: one fractional A100, ~5 minutes.
#
# Submit:
#   sbatch experiments/slurm/saliency.sbatch
#   sbatch --export=CHECKPOINT=results/lora_seed0/best.pt,N_SAMPLES=6 \
#          experiments/slurm/saliency.sbatch

#SBATCH --job-name=omnict-saliency
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100_1g.5gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=results/slurm-%x-%j.out
#SBATCH --error=results/slurm-%x-%j.err

set -euo pipefail

: "${CONFIG:=src/configs/lora.yaml}"
: "${CHECKPOINT:=results/lora_seed0/best.pt}"
: "${N_SAMPLES:=6}"
: "${OUT:=report/figures/saliency_panel.pdf}"

cd "${SLURM_SUBMIT_DIR}"

module purge || true
module load python
source .venv/bin/activate

python -m src.explain.run_saliency \
    --config "${CONFIG}" \
    --checkpoint "${CHECKPOINT}" \
    --n_samples "${N_SAMPLES}" \
    --out "${OUT}"

echo "[saliency] wrote ${OUT}"
