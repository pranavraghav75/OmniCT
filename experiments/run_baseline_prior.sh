#!/usr/bin/env bash
# Reproduce Table 1, Row 1 — class-prior baseline.
# Trivial baseline: outputs the empirical training-set class prior.
# 3 seeds for variance.
set -euo pipefail

CFG=src/configs/baseline_prior.yaml

for SEED in 0 1 2; do
  python -m src.training.train \
    --config "${CFG}" \
    --override "seed=${SEED}" \
    --run_name "prior_seed${SEED}"
done
