# Data

This directory holds download scripts describing the splits we use. The following should be the
layout after running the donload script:

```
data/
├── README.md
├── download_flare.py          # downloads a subset from HuggingFace
├── manifests/
│   ├── labels.csv             # volume_id, organ, label (0=benign, 1=malignant)
│   ├── splits/
│   │   ├── train.txt
│   │   ├── val.txt
│   │   └── test.txt
├── raw/                       # NIfTI volumes (gitignored)
│   ├── <volume_id>.nii.gz
│   └── ...
└── processed/                 # cached preprocessed tensors (gitignored)
    └── <volume_id>.pt
```

We pull from the CVPR 2026 General CT Image Diagnosis workshop:

- `FLARE-MedFM/FLARE-Task4-CT-FM` — pretraining set, CT volumes
- `kmin06/CVPR26-3DCTFMCompetition` — classification labels for the downstream task

For the project, we use a stratified subset (default: 1000 volumes total, ~50/50 class split where possible, capped per-organ to avoid any bias). Download found below:

```bash
python data/download_flare.py \
    --out data/raw \
    --manifest data/manifests/labels.csv \
    --max_volumes 1000 \
    --seed 0
```

The donwload_flare.py script has the following functionality:
1. Streams the HuggingFace dataset metadata
2. Selects a stratified subset
3. Downloads each NIfTI to `data/raw/`
4. Writes `manifests/labels.csv` and the train/val/test split files

Re-running with the same `--seed` produces the same subset. We use a patient-disjoint 70/15/15 train/val/test split (no patient leakage across splits). The split is stratified by organ + label.

To regenerate splits from the labels CSV:

```bash
python -m src.data.make_splits \
    --labels data/manifests/labels.csv \
    --out_dir data/manifests/splits \
    --seed 0
```