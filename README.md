# OmniCT

**One-line description**: Multi-organ malignancy classification from 3D CT volumes via parameter-efficient fine-tuning of pretrained 3D medical foundation models.

CMSC 472 (Spring 2026) Final Project — Emily Wang, Annika Wong, Alec Zhou, Pranav Raghavan.

---

## 1. Project at a glance

CT scans must be read by radiologists, and interpretive errors increase with shift length and scan volume (Hanna et al., 2018). We build **OmniCT**, a system that takes a 3D CT volume and predicts whether a malignant lesion is present, generalizing across multiple organ systems rather than being tied to a single anatomical region (as Merlin or 3DINO often are).

Our approach:

1. **Standardized preprocessing** — Hounsfield Unit (HU) clipping and isotropic resampling (per nnU-Net conventions, Isensee et al. 2021).
2. **Frozen 3D foundation-model encoder** (3DINO / SPECTRE / similar) producing a volume-level embedding.
3. **Parameter-Efficient Fine-Tuning** — a lightweight LoRA-adapted classification head trained for binary malignancy classification.
4. **Baselines** — class-prior, small from-scratch 3D CNN, and frozen-feature linear probe (à la Pai et al. 2024).

Reported metrics: ROC-AUC, F1, Balanced Accuracy, mean ± std over 3 seeds.

## 2. Repository layout

```
OmniCT/
├── README.md              # this file
├── requirements.txt       # pinned deps
├── data/                  # download scripts + manifests (no raw volumes committed)
│   ├── README.md
│   └── download_flare.py
├── src/                   # library code
│   ├── configs/           # YAML configs (default + per-experiment overrides)
│   ├── data/              # preprocessing transforms + torch Dataset
│   ├── models/            # baselines, foundation-model wrappers, LoRA head
│   ├── training/          # train loop, eval, metrics
│   └── utils/             # seeding, config loading, logging
├── experiments/           # shell scripts to reproduce each main result
├── results/               # logs, figures, tables (checkpoints gitignored)
└── notebooks/             # optional exploratory notebooks
```

## 3. Setup

```bash
# 1. Clone
git clone https://github.com/pranavraghav75/OmniCT.git
cd OmniCT

# 2. Create env (python >= 3.10 recommended)
python -m venv .venv
source .venv/bin/activate

# 3. Install
pip install --upgrade pip
pip install -r requirements.txt
```

GPU is strongly recommended. The pipeline runs CPU-only for smoke tests but is impractical there for real training.

## 4. Data

We use a subset of the CVPR 2026 General CT Image Diagnosis workshop datasets:

- [FLARE-Task4-CT-FM](https://huggingface.co/datasets/FLARE-MedFM/FLARE-Task4-CT-FM)
- [CVPR26-3DCTFMCompetition](https://huggingface.co/datasets/kmin06/CVPR26-3DCTFMCompetition)

To download the subset we use:

```bash
python data/download_flare.py --out data/raw --max_volumes 1000
```

This produces `data/raw/<volume_id>.nii.gz` files plus `data/manifests/labels.csv` (columns: `volume_id,organ,label`). See `data/README.md` for full instructions.

## 5. Reproducing main results

Each command below writes to a separate sub-directory under `results/` and logs metrics to `results/<run_name>/metrics.json`.

```bash
# Table 1, row 1 — class-prior baseline
bash experiments/run_baseline_prior.sh

# Table 1, row 2 — small 3D CNN trained from scratch
bash experiments/run_baseline_cnn3d.sh

# Table 1, row 3 — frozen-feature linear probe
bash experiments/run_linear_probe.sh

# Table 1, row 4 — LoRA-adapted foundation model (main method)
bash experiments/run_lora.sh

# Figure 2 — diagnostic / analysis experiment (data-efficiency curve)
bash experiments/run_data_efficiency.sh

# All of the above (3 seeds each)
bash experiments/run_all.sh
```

After the runs complete:

```bash
python -m src.training.aggregate_results --runs_dir results/ --out results/tables/main.csv
python -m src.training.make_figures      --runs_dir results/ --out results/figures/
```

## 6. Expected runtime & hardware

| Experiment            | Hardware         | Wall-clock (approx) |
|-----------------------|------------------|---------------------|
| Smoke test (CPU)      | any laptop       | ~2 min              |
| 3D CNN baseline       | 1 × A100 40GB    | ~45 min / seed      |
| Linear probe          | 1 × A100 40GB    | ~20 min / seed      |
| LoRA fine-tune        | 1 × A100 40GB    | ~1.5 hr / seed      |
| Full reproduction (3 seeds, all rows) | 1 × A100 | ~10 hr |

## 7. Reproducibility

- All scripts accept `--seed` (defaults to the `seed:` field of the YAML config).
- `src/utils/seeding.py` seeds `random`, `numpy`, and `torch` (CPU + CUDA), and sets `torch.backends.cudnn.deterministic=True`.
- Each run writes its full resolved config to `results/<run_name>/config.yaml` so a re-run with the same config + seed reproduces results.

## 8. AI / LLM tool disclosure

Per the course policy, we disclose:

- **Coding assistance**: portions of the codebase scaffolding (directory layout, boilerplate training loop, config plumbing) were drafted with the help of an LLM-based coding assistant. All code was reviewed and tested by team members; we are responsible for its correctness.
- **Report**: all prose in the final report is written by the team.

## 9. License

For class-project use only. Underlying datasets retain their original licenses.

## 10. Citation

If you build on this code, please cite the CVPR 2026 FMV workshop and the foundation-model papers we adapt (3DINO, SPECTRE, Merlin, Pai et al. 2024) — see `report/references.bib` for full BibTeX entries.
