#!/usr/bin/env bash
# Reproduce Figure (saliency panel) — applies gradient-times-input and
# 3D Grad-CAM to a small set of validation volumes using the best
# checkpoint of each method, and saves a single PDF for the report.
set -euo pipefail

mkdir -p report/figures

# Use the best LoRA seed=0 checkpoint by default. Override via env vars.
: "${CONFIG:=src/configs/lora.yaml}"
: "${CHECKPOINT:=results/lora_seed0/best.pt}"
: "${N_SAMPLES:=6}"
: "${OUT:=report/figures/saliency_panel.pdf}"

python -m src.explain.run_saliency \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --n_samples "${N_SAMPLES}" \
  --out "${OUT}"

echo "Saved saliency panel to ${OUT}"
