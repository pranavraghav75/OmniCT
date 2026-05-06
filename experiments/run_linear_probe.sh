#!/usr/bin/env bash
# Reproduce Table 1, Row 3 — frozen 3D foundation model + linear probe.
# (Pai et al. 2024 style: linear classifier on top of a frozen pretrained encoder.)
set -euo pipefail

CFG=src/configs/linear_probe.yaml

for SEED in 0 1 2; do
  python -m src.training.train \
    --config "${CFG}" \
    --override "seed=${SEED}" \
    --run_name "linear_probe_seed${SEED}"
done
