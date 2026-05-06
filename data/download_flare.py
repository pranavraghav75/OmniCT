"""Download a stratified subset of the CVPR26 / FLARE-Task4 CT-FM data.

This script streams the HuggingFace dataset listing, picks a deterministic stratified subset,
downloads each NIfTI to ``--out`` and writes ``data/manifests/labels.csv``.

We do not pull the entire 10k-volume corpus as it would be impractical and unnecessary 
for the experiments we plan to run.

Usage:
    python data/download_flare.py \
        --out data/raw \
        --manifest data/manifests/labels.csv \
        --max_volumes 1000 \
        --seed 0

NOTE: the exact field names below depend on the HuggingFace dataset schema which
we resolve at first run by introspecting `dataset.column_names`. Update
`LABEL_COLUMN` / `ORGAN_COLUMN` if the schema differs from what we
checked at proposal time.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

def _lazy_imports():
    from datasets import load_dataset  
    from huggingface_hub import hf_hub_download 
    return load_dataset, hf_hub_download


DATASET_REPO = "kmin06/CVPR26-3DCTFMCompetition"
LABEL_COLUMN = "label"
ORGAN_COLUMN = "organ"
PATH_COLUMN = "image"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("data/raw"))
    p.add_argument("--manifest", type=Path, default=Path("data/manifests/labels.csv"))
    p.add_argument("--max_volumes", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Skip the actual file copy; just write the manifest.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    load_dataset, _ = _lazy_imports()

    args.out.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset metadata: {DATASET_REPO}")
    ds = load_dataset(DATASET_REPO, split="train")
    print(f"  total rows: {len(ds)}")
    print(f"  columns: {ds.column_names}")

    for col in (LABEL_COLUMN, PATH_COLUMN):
        if col not in ds.column_names:
            raise SystemExit(
                f"Expected column {col!r} not found. "
                f"Available: {ds.column_names}. "
                f"Edit LABEL_COLUMN / PATH_COLUMN at the top of this script."
            )

    ds = ds.shuffle(seed=args.seed)
    if args.max_volumes is not None and args.max_volumes < len(ds):
        ds = ds.select(range(args.max_volumes))

    rows = []
    for i, row in enumerate(ds):
        src = row[PATH_COLUMN]
        if hasattr(src, "path"):
            src_path = Path(src.path)
        else:
            src_path = Path(src)

        ext = "".join(src_path.suffixes) or ".nii.gz"
        volume_id = f"vol_{i:06d}"
        dst = args.out / f"{volume_id}{ext}"

        if not args.dry_run and not dst.exists():
            shutil.copy(src_path, dst)

        rows.append(
            {
                "volume_id": volume_id,
                "path": str(dst.relative_to(args.out.parent)) if dst.is_absolute() else str(dst),
                "label": int(row[LABEL_COLUMN]),
                "organ": str(row.get(ORGAN_COLUMN, "unknown")),
            }
        )

    with args.manifest.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["volume_id", "path", "label", "organ"])
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} rows -> {args.manifest}")
    print(f"Volumes -> {args.out}")


if __name__ == "__main__":
    main()
