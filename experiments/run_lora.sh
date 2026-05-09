#!/usr/bin/env bash
# LoRA-style adaptation / fine-tuning (3 seeds).
set -euo pipefail

CFG=src/configs/lora.yaml

for SEED in 0 1 2; do
  python -m src.training.train \
    --config "${CFG}" \
    --override "seed=${SEED}" \
              "data.synthetic=false" \
              "data.spatial_size=[32,32,32]" \
              "data.spacing=[1.0,1.0,1.0]" \
              "data.hu_window=[-1000.0,400.0]" \
    --run_name "lora_seed${SEED}"
done
