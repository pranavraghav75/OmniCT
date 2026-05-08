#!/usr/bin/env bash
# Tiny CPU smoke test — verifies the whole pipeline (data, model,
# training loop, metrics) wires up end-to-end on synthetic data.
# Should finish in <1 minute on any laptop.
set -euo pipefail

python -m src.training.train \
  --config src/configs/baseline_cnn3d.yaml \
  --override "seed=0" "device=cpu" "train.epochs=1" \
              "data.synthetic_n_train=16" "data.synthetic_n_val=8" \
              "data.spatial_size=[32,32,32]" "train.batch_size=2" "train.num_workers=0" \
  --run_name smoke_test

echo
echo "Smoke test passed. Outputs in results/smoke_test/"
