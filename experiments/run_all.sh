#!/usr/bin/env bash
# Run every experiment used in the paper (Table 1 + Figure 2).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

bash "${HERE}/run_baseline_prior.sh"
bash "${HERE}/run_baseline_cnn3d.sh"
bash "${HERE}/run_linear_probe.sh"
bash "${HERE}/run_lora.sh"
bash "${HERE}/run_data_efficiency.sh"

# Aggregate per-run metrics.json into one table.
python -m src.training.aggregate_results --runs_dir results --out results/tables/main.csv
