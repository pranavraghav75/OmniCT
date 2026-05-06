#!/usr/bin/env bash
# Reproduce Table 1, Row 2 — small 3D CNN trained from scratch.
set -euo pipefail

CFG=src/configs/baseline_cnn3d.yaml

for SEED in 0 1 2; do
  python -m src.training.train \
    --config "${CFG}" \
    --override "seed=${SEED}" \
    --run_name "cnn3d_seed${SEED}"
done
