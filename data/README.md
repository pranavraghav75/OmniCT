# Data

This directory holds dataset download scripts and manifests. After running a download script, the
layout should look like:

```
data/
├── README.md
├── download_flare.py          # downloads a subset from HuggingFace
├── download_medmnist.py       # downloads NoduleMNIST3D (public; recommended)
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

## Recommended dataset: NoduleMNIST3D

For the final report runs we use **NoduleMNIST3D**, a public 3D CT lung nodule malignancy
benchmark (small, quick to download, no auth). Download + materialize the dataset with:

```bash
python data/download_medmnist.py --out data/raw
```

This writes:

- `data/raw/*.nii.gz`
- `data/manifests/labels.csv`
- `data/manifests/splits/{train,val,test}.txt`

## Optional / legacy: CVPR'26 workshop subset

Earlier iterations attempted to pull from the CVPR 2026 General CT Image Diagnosis workshop:

- `FLARE-MedFM/FLARE-Task4-CT-FM` — pretraining set, CT volumes
- `kmin06/CVPR26-3DCTFMCompetition` — classification labels for the downstream task

We keep `download_flare.py` for reference, but the final report does not depend on it.

```bash
python data/download_flare.py \
    --out data/raw \
    --manifest data/manifests/labels.csv \
    --max_volumes 1000 \
    --seed 0
```

The `download_flare.py` script:
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