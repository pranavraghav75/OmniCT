#!/usr/bin/env bash
# Reproduce Table 1, Row 4 — main method.
# 3D foundation encoder with LoRA-injected linear layers + MLP head.
set -euo pipefail

CFG=src/configs/lora.yaml

for SEED in 0 1 2; do
  python -m src.training.train \
    --config "${CFG}" \
    --override "seed=${SEED}" \
    --run_name "lora_seed${SEED}"
done
