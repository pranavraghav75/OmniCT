# OmniCT

**One-line description**: Lung nodule malignancy classification from 3D CT volumes, comparing simple baselines and parameter-efficient adaptation.

CMSC 472 (Spring 2026) Final Project — Emily Wang, Annika Wong, Alec Zhou, Pranav Raghavan.

---

## 1. Project at a glance

CT scans must be read by radiologists, and interpretive errors increase with shift length and scan volume (Hanna et al., 2018). We build **OmniCT**, a small, reproducible study that takes a 3D CT volume and predicts whether a lung nodule is malignant.

Our approach:

1. **Preprocessing** — HU windowing to \([-1000, 400]\), normalization, and resizing.
2. **Baselines** — class-prior, a small from-scratch 3D CNN, and a frozen-feature linear probe.
3. **Adaptation** — a LoRA-style recipe; when the chosen backbone has no matching Linear modules, we fall back to end-to-end fine-tuning (documented in code and paper).

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
│   ├── explain/           # saliency / 3D Grad-CAM (localization)
│   └── utils/             # seeding, config loading, logging
├── experiments/           # shell scripts to reproduce each main result
├── report/                # NeurIPS LaTeX paper source + figures + tables
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

We use **NoduleMNIST3D** (public, small, no auth). Download and materialize the dataset into the project layout with:

```bash
python data/download_medmnist.py --out data/raw
```

This produces `data/raw/<volume_id>.nii.gz` plus `data/manifests/labels.csv` and split files. See `data/README.md`.

## 5. Reproducing main results

Each script below writes to `results/<run_name>/` and logs metrics to `results/<run_name>/metrics.json`.

```bash
# Table 1, row 1 — class-prior baseline
bash experiments/run_baseline_prior.sh  # requires data download

# Table 1, row 2 — small 3D CNN trained from scratch
bash experiments/run_baseline_cnn3d.sh  # requires data download

# Table 1, row 3 — frozen-feature linear probe
bash experiments/run_linear_probe.sh    # requires data download

# Table 1, row 4 — LoRA-adapted foundation model (main method)
bash experiments/run_lora.sh            # requires data download

# Figure 2 — diagnostic / analysis experiment (data-efficiency curve)
bash experiments/run_data_efficiency.sh # optional

# Figure 3 — saliency / 3D Grad-CAM panel (localization, addresses
# the project-feedback recommendation to provide a localization signal)
bash experiments/run_saliency.sh

# All of the above (3 seeds each)
bash experiments/run_all.sh
```

After the runs complete:

```bash
python -m src.training.aggregate_results --runs_dir results/ --out results/tables/main.csv
```

## 6. Expected runtime & hardware (Colab)

Typical end-to-end runtime for the full grid (3 seeds, 4 methods) is **~20–40 minutes on a Colab A100**, plus a few minutes for saliency generation.

## 6.1 Colab

We provide a Colab notebook that runs the full pipeline end-to-end:

- `notebooks/colab_train.ipynb`

## 6.1 Report

The NeurIPS-formatted final report lives in `report/`. All 7 required
sections (Intro, Related Work, Method, Experiments, Analysis,
Discussion, Conclusion) are pre-stubbed with content seeded from the
project proposal; search `\TODO{}` in `report/main.tex` to find every
remaining drafting task. Build with `make -C report` (requires
`pdflatex` + `bibtex`).

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
