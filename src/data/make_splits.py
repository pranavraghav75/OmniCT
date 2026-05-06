"""Generate patient-disjoint, stratified train/val/test splits.

Run:
    python -m src.data.make_splits \
        --labels data/manifests/labels.csv \
        --out_dir data/manifests/splits \
        --seed 0 \
        --train 0.7 --val 0.15 --test 0.15

We split on `volume_id` (treated as a proxy for patient) so a single
patient never appears in more than one split.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--labels", required=True, type=Path)
    p.add_argument("--out_dir", required=True, type=Path)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train", type=float, default=0.7)
    p.add_argument("--val", type=float, default=0.15)
    p.add_argument("--test", type=float, default=0.15)
    args = p.parse_args()

    if abs(args.train + args.val + args.test - 1.0) > 1e-6:
        raise SystemExit("train + val + test must sum to 1")

    df = pd.read_csv(args.labels)
    rng = np.random.default_rng(args.seed)

    # Stratify by (organ, label) to keep class balance per split.
    if "organ" in df.columns:
        strata = df["organ"].astype(str) + "::" + df["label"].astype(str)
    else:
        strata = df["label"].astype(str)

    train_ids: list[str] = []
    val_ids: list[str] = []
    test_ids: list[str] = []

    for _, group in df.groupby(strata):
        ids = group["volume_id"].tolist()
        rng.shuffle(ids)
        n = len(ids)
        n_train = int(round(n * args.train))
        n_val = int(round(n * args.val))
        train_ids.extend(ids[:n_train])
        val_ids.extend(ids[n_train : n_train + n_val])
        test_ids.extend(ids[n_train + n_val :])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        out = args.out_dir / f"{name}.txt"
        out.write_text("\n".join(ids) + "\n")
        print(f"  wrote {len(ids):>6d} ids -> {out}")


if __name__ == "__main__":
    main()
