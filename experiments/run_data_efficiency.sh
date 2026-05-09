#!/usr/bin/env bash
# Data-efficiency experiment (optional).
set -euo pipefail

CFG=src/configs/lora.yaml

for FRAC in 0.10 0.25 0.50 1.00; do
  for SEED in 0 1 2; do
    python -m src.training.train \
      --config "${CFG}" \
      --override "seed=${SEED}" \
                "data.synthetic=false" \
                "data.spatial_size=[32,32,32]" \
                "data.spacing=[1.0,1.0,1.0]" \
                "data.hu_window=[-1000.0,400.0]" \
                "data.train_frac_subsample=${FRAC}" \
      --run_name "lora_frac${FRAC}_seed${SEED}"
  done
done
