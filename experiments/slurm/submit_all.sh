#!/usr/bin/env bash
# Submits the entire OmniCT experiment grid to Zaratan in one shot:
#   - 4 methods (prior / cnn3d / linear_probe / lora)
#   - 3 seeds each, run in parallel via job arrays
#   - Aggregation job that runs after all training finishes (Slurm
#     dependency).
#
# Run from the project root on a Zaratan login node:
#   bash experiments/slurm/submit_all.sh
set -euo pipefail

CONFIGS=(
  "src/configs/baseline_prior.yaml"
  "src/configs/baseline_cnn3d.yaml"
  "src/configs/linear_probe.yaml"
  "src/configs/lora.yaml"
)

# Append "data.synthetic=false" once you've downloaded real data.
EXTRA="${EXTRA_OVERRIDES:-}"

JOB_IDS=()
for CFG in "${CONFIGS[@]}"; do
  echo "[submit] $CFG"
  JID=$(sbatch --parsable \
               --export=ALL,CONFIG="${CFG}",EXTRA_OVERRIDES="${EXTRA}" \
               experiments/slurm/train_array.sbatch)
  echo "         array job id = $JID"
  JOB_IDS+=("$JID")
done

# Build a colon-separated list of job IDs for the aggregation job's
# afterok dependency.
DEP=$(IFS=:; echo "${JOB_IDS[*]}")

echo "[submit] aggregation job (depends on: $DEP)"
AGG_JID=$(sbatch --parsable \
                 --dependency=afterok:"$DEP" \
                 experiments/slurm/aggregate.sbatch)
echo "         agg job id   = $AGG_JID"

echo
echo "Track progress with:    squeue --me"
echo "Inspect a job:          scontrol show job <jid>"
echo "Cancel everything:      scancel ${JOB_IDS[*]} $AGG_JID"
