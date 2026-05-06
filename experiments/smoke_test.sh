#!/usr/bin/env bash
# Tiny CPU smoke test — verifies the whole pipeline (data, model,
# training loop, metrics) wires up. Should finish in a couple of minutes
# on any laptop. Useful before kicking off real GPU runs.
set -euo pipefail

python -m src.training.train \
  --config src/configs/baseline_cnn3d.yaml \
  --override seed=0 device=cpu train.epochs=1 \
              data.synthetic_n_train=16 data.synthetic_n_val=8 \
              data.spatial_size=[32,32,32] train.batch_size=2 \
  --run_name smoke_test
