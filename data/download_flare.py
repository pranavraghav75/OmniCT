"""(Legacy) Download a LUNA25 subset from the CVPR'26 workshop dataset.

We use ``huggingface_hub.hf_hub_download`` rather than
``datasets.load_dataset`` because the upstream repository mixes
multiple file formats across splits (NIfTI folders + CSV labels), which
``load_dataset`` cannot auto-infer. Pulling files directly is simpler
and works.

The labels CSV (``LUNA25/labels/lung_nodule_malignancy.csv``) has
columns:

    case_id, lung_nodule_malignancy, split

where ``case_id`` is the NIfTI filename, ``lung_nodule_malignancy`` is
0 (benign) or 1 (malignant), and ``split`` is ``train`` or ``val``.

We translate this into the project-standard manifest:

    volume_id, path, label, organ

with ``organ='lung'`` for every row.

Usage:
    # Pull the full LUNA25 subset (~1200 volumes, ~30 GB):
    python data/download_flare.py --out data/raw

    # Stratified subsample to keep things manageable:
    python data/download_flare.py --out data/raw --max_volumes 400 --balance

    # Just get the labels CSV first to inspect the manifest:
    python data/download_flare.py --out data/raw --no_volumes

    # Plan the download without touching the network:
    python data/download_flare.py --out data/raw --max_volumes 50 --dry_run
"""

from __future__ import annotations

import argparse
import csv
import io
import shutil
import sys
from pathlib import Path
from typing import Iterable

DATASET_REPO = "kmin06/CVPR26-3DCTFMCompetition"
LABELS_PATH_IN_REPO = "LUNA25/labels/lung_nodule_malignancy.csv"
IMAGES_DIR_IN_REPO = "LUNA25/images"
ORGAN = "lung"


def _hf_imports():
    from huggingface_hub import hf_hub_download  # type: ignore
    return hf_hub_download


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("data/raw"))
    p.add_argument(
        "--manifest", type=Path, default=Path("data/manifests/labels.csv")
    )
    p.add_argument(
        "--max_volumes",
        type=int,
        default=None,
        help="If set, keep only this many volumes (deterministic, seeded).",
    )
    p.add_argument(
        "--balance",
        action="store_true",
        help="When subsampling, keep the class ratio close to 50/50.",
    )
    p.add_argument(
        "--no_volumes",
        action="store_true",
        help="Skip downloading volumes; just write the manifest from the labels CSV.",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Plan the work and write the manifest, but skip every network download.",
    )
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _stratified_subsample(
    rows: list[dict], max_volumes: int, balance: bool, seed: int
) -> list[dict]:
    """Return up to `max_volumes` rows, optionally rebalanced 50/50 per class."""
    import random

    rng = random.Random(seed)
    if not balance:
        rng.shuffle(rows)
        return rows[:max_volumes]

    by_class: dict[str, list[dict]] = {"0": [], "1": []}
    for r in rows:
        by_class.setdefault(r["label"], []).append(r)
    for v in by_class.values():
        rng.shuffle(v)

    per_class = max_volumes // 2
    out = (
        by_class.get("0", [])[:per_class]
        + by_class.get("1", [])[:per_class]
    )
    rng.shuffle(out)
    return out


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    hf_hub_download = _hf_imports()

    # Step 1: pull the labels CSV (~92 KB).
    print(f"[download] fetching labels: {LABELS_PATH_IN_REPO}")
    labels_local = hf_hub_download(
        repo_id=DATASET_REPO,
        filename=LABELS_PATH_IN_REPO,
        repo_type="dataset",
    )
    print(f"           -> {labels_local}")

    # Step 2: parse it.
    rows: list[dict] = []
    with open(labels_local, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            case_id = r["case_id"].strip()
            label = r["lung_nodule_malignancy"].strip()
            hf_split = r["split"].strip()
            rows.append(
                {
                    "case_id": case_id,
                    "label": label,
                    "hf_split": hf_split,
                }
            )
    print(f"[download] parsed {len(rows)} rows from labels CSV")
    counts = {("0", "train"): 0, ("0", "val"): 0, ("1", "train"): 0, ("1", "val"): 0}
    for r in rows:
        counts[(r["label"], r["hf_split"])] = counts.get((r["label"], r["hf_split"]), 0) + 1
    print(f"           label/split breakdown: {counts}")

    # Step 3: optional subsampling.
    if args.max_volumes is not None and args.max_volumes < len(rows):
        rows = _stratified_subsample(
            rows, args.max_volumes, balance=args.balance, seed=args.seed
        )
        print(f"[download] subsampled to {len(rows)} rows (balance={args.balance})")

    # Step 4: download each volume.
    manifest_rows: list[dict] = []
    n_skip = 0
    for i, r in enumerate(rows):
        case = r["case_id"]
        repo_path = f"{IMAGES_DIR_IN_REPO}/{case}"
        local_dst = args.out / case

        if not args.no_volumes and not args.dry_run and not local_dst.exists():
            try:
                src = hf_hub_download(
                    repo_id=DATASET_REPO,
                    filename=repo_path,
                    repo_type="dataset",
                )
                # huggingface_hub stores files in a content-addressed cache
                # and returns a (possibly symlinked) path; copy to our `out`
                # so the project layout doesn't depend on the cache.
                shutil.copy(src, local_dst)
            except Exception as e:
                print(f"  [warn] failed to fetch {case}: {e}", file=sys.stderr)
                n_skip += 1
                continue

        manifest_rows.append(
            {
                "volume_id": case.removesuffix(".nii.gz"),
                "path": str(local_dst.relative_to(args.out.parent)),
                "label": int(r["label"]),
                "organ": ORGAN,
            }
        )

        if (i + 1) % 25 == 0:
            print(f"  [progress] {i + 1}/{len(rows)} (skipped: {n_skip})")

    # Step 5: write the manifest.
    with args.manifest.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["volume_id", "path", "label", "organ"])
        w.writeheader()
        w.writerows(manifest_rows)

    print()
    print(f"[done] wrote manifest -> {args.manifest}  ({len(manifest_rows)} rows)")
    if not args.dry_run and not args.no_volumes:
        print(f"[done] volumes in    -> {args.out}/")
    if args.dry_run:
        print("[done] (dry run; no volumes were downloaded)")


if __name__ == "__main__":
    main()
