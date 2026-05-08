# Running OmniCT on UMD Zaratan

End-to-end SLURM workflow. Assumes you have a UMD Linux account + an
active Zaratan allocation (free Developmental, class allocation, or
research-group allocation).

## 0. Prerequisites

- UMD Directory ID (same as your email login).
- An active SLURM allocation. Check with `sbalance` after logging in.
- UMD VPN (GlobalProtect) connected, *or* be ready to do Duo MFA on
  every SSH attempt.

## 1. SSH in

```bash
# From your laptop, with the UMD VPN connected:
ssh <yourdirID>@login.zaratan.umd.edu
```

## 2. One-time setup (run on a login node)

```bash
# Pull the project + install all deps, including a CUDA build of torch.
# This puts the venv on /scratch (NOT $HOME, which has a small quota).
curl -fsSL https://raw.githubusercontent.com/pranavraghav75/OmniCT/main/experiments/slurm/setup_zaratan.sh \
    | bash
```

If you'd rather see what's about to happen first:

```bash
git clone https://github.com/pranavraghav75/OmniCT.git ~/scratch.gw/$USER/OmniCT
cd ~/scratch.gw/$USER/OmniCT
bash experiments/slurm/setup_zaratan.sh
```

The setup script:
1. Clones the repo into `~/scratch.gw/$USER/OmniCT`.
2. Loads the `python` module.
3. Creates a `.venv/` and installs a CUDA-enabled `torch` (cu121
   wheels) plus the rest of `requirements.txt`.
4. Runs the CPU smoke test to verify wiring.

## 3. GPU smoke test (10 min, debug partition, ~1 SU)

Confirms a real GPU node can pick up your venv, find CUDA, and run the
training loop end-to-end.

```bash
cd ~/scratch.gw/$USER/OmniCT
sbatch experiments/slurm/smoke.sbatch
squeue --me                 # watch it move from PD -> R -> done
```

Output appears in `results/slurm-omnict-smoke-<jobid>.out`.

## 4. Submit the full experiment grid

```bash
bash experiments/slurm/submit_all.sh
```

This fires off four independent array jobs (one per method), each with
3 seeds running in parallel on separate A100s, plus an aggregation job
that depends on all of them and writes `results/tables/main.csv`.

Track progress:

```bash
squeue --me                                      # what's queued / running
sacct -X --starttime=today --format=JobID,JobName,State,Elapsed
tail -f results/slurm-omnict-array-<jobid>_0.err # live log of seed 0
```

Cancel a misbehaving job: `scancel <jobid>`.

## 5. After training finishes

```bash
# Pull the saliency figure
sbatch --export=CHECKPOINT=results/lora_seed0/best.pt \
       experiments/slurm/saliency.sbatch    # see ./saliency.sbatch

# Or pull results back to your laptop and finish locally:
rsync -avz --exclude '*.pt' \
    <dirID>@login.zaratan.umd.edu:~/scratch.gw/<dirID>/OmniCT/results/ \
    ./results-from-zaratan/
```

## Tips

- **Login nodes are not for compute.** Do not run `python ...` on a
  login node — IT will kill it and warn you. Always `sbatch` or use
  `salloc` for an interactive GPU shell.
- **Use `salloc -p gpu --gres=gpu:a100:1 -t 1:00:00 --pty bash`** for
  an interactive GPU session if you want to debug live.
- **Storage**: `$HOME` is small (10 GB-ish). Keep code, env, and data
  on `$HOME/scratch.gw/$USER/...`. The setup script does this for you.
- **Allocation cost**: a full A100 GPU costs **48 SU/hour**; the full
  experiment grid (4 methods × 3 seeds × ~1.5 hr) is roughly
  **300–400 SU**. Free Developmental Allocations are 50000 SU/quarter,
  so you have headroom.
- **If torch can't see CUDA** on a GPU node, try replacing
  `module load python` with `module load python cuda/12.1` in the
  sbatch file, or rebuild the venv pointing at a different cu wheel
  (cu118 / cu124).
