#!/usr/bin/env bash
# Reproduce Figure 2 — data-efficiency curve (the analysis experiment
# required by the rubric). For each training-set fraction f in {0.1, 0.25,
# 0.5, 1.0}, retrain LoRA from scratch and report test ROC-AUC. Useful to
# show whether the foundation backbone helps in low-data regimes.
set -euo pipefail

CFG=src/configs/lora.yaml

for FRAC in 0.10 0.25 0.50 1.00; do
  for SEED in 0 1 2; do
    python -m src.training.train \
      --config "${CFG}" \
      --override "seed=${SEED}" "data.synthetic_n_train=$(python -c "print(int(128*${FRAC}))")" \
      --run_name "lora_frac${FRAC}_seed${SEED}"
  done
done
