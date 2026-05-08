#!/usr/bin/env bash
# Aggregates per-seed metrics.json files into Table 1 of the paper.
# Runs on a CPU node; doesn't need a GPU.
#
# Usually submitted automatically by submit_all.sh as an afterok
# dependency; you can also run it manually any time.

#SBATCH --job-name=omnict-aggregate
#SBATCH --partition=standard
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=results/slurm-%x-%j.out
#SBATCH --error=results/slurm-%x-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p results/tables

module purge || true
module load python
source .venv/bin/activate

python -m src.training.aggregate_results \
    --runs_dir results \
    --out results/tables/main.csv \
    --split test

echo "[aggregate] wrote results/tables/main.csv"
cat results/tables/main.csv
