"""Download NoduleMNIST3D and write it into the project's layout.

Writes NIfTI volumes into `data/raw/` plus:
  - `data/manifests/labels.csv`
  - `data/manifests/splits/{train,val,test}.txt`
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np


HU_LO, HU_HI = -1000.0, 400.0


def _imports():
    try:
        from medmnist import NoduleMNIST3D, INFO  # type: ignore
    except Exception as e:
        raise SystemExit(
            "[error] could not import `medmnist`. Install it first:\n"
            "    pip install medmnist\n"
            f"  underlying error: {e}"
        )
    try:
        import nibabel as nib
    except Exception as e:
        raise SystemExit(
            "[error] could not import `nibabel`. Install it first:\n"
            "    pip install nibabel\n"
            f"  underlying error: {e}"
        )
    return NoduleMNIST3D, INFO, nib


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("data/raw"))
    p.add_argument(
        "--manifest", type=Path, default=Path("data/manifests/labels.csv")
    )
    p.add_argument(
        "--splits_dir", type=Path, default=Path("data/manifests/splits")
    )
    p.add_argument(
        "--max_per_split",
        type=int,
        default=None,
        help=(
            "If set, keep only this many volumes per split (deterministic, "
            "seeded). Useful for fast smoke runs."
        ),
    )
    p.add_argument(
        "--balance",
        action="store_true",
        help="When subsampling, keep an even class ratio per split.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--size", type=int, default=28,
        help="Voxel size for the dataset variant. 28 is the original; "
             "MedMNIST also offers 64 for some splits.",
    )
    return p.parse_args()


def _to_hu(volume_uint8: np.ndarray) -> np.ndarray:
    """Map uint8 [0, 255] linearly to fake HU [-1000, 400] (float32)."""
    v = volume_uint8.astype(np.float32) / 255.0
    return (HU_LO + v * (HU_HI - HU_LO)).astype(np.float32)


def _stratified_sub(
    indices: List[int], labels: List[int], k: int, balance: bool, seed: int
) -> List[int]:
    rng = np.random.default_rng(seed)
    if not balance:
        idx = list(indices)
        rng.shuffle(idx)
        return idx[:k]
    pos = [i for i, y in zip(indices, labels) if y == 1]
    neg = [i for i, y in zip(indices, labels) if y == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)
    per = k // 2
    out = pos[:per] + neg[:per]
    rng.shuffle(out)
    return out


def main() -> None:
    args = parse_args()
    NoduleMNIST3D, INFO, nib = _imports()

    args.out.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.splits_dir.mkdir(parents=True, exist_ok=True)

    print("[medmnist] info:", INFO["nodulemnist3d"]["task"],
          "n_classes=", INFO["nodulemnist3d"]["n_channels"],
          "label=", INFO["nodulemnist3d"]["label"])

    splits_to_load = [("train", "train"), ("val", "val"), ("test", "test")]
    manifest_rows: list[dict] = []
    split_to_ids: dict[str, list[str]] = {"train": [], "val": [], "test": []}

    counter = 0
    for hf_split, our_split in splits_to_load:
        ds = NoduleMNIST3D(split=hf_split, download=True, size=args.size)
        x_all = ds.imgs            
        y_all = np.array(ds.labels).reshape(-1).astype(int).tolist()
        if x_all.ndim == 5:        
            x_all = x_all[:, 0]
        n = len(x_all)
        idx_pool = list(range(n))

        if args.max_per_split is not None and args.max_per_split < n:
            idx_pool = _stratified_sub(
                idx_pool, y_all, args.max_per_split, args.balance, args.seed
            )
            print(f"[medmnist] {our_split}: subsampled {len(idx_pool)}/{n}"
                  f" (balance={args.balance})")
        else:
            print(f"[medmnist] {our_split}: keeping all {n} volumes")

        for j, k in enumerate(idx_pool):
            vid = f"nodule_{counter:05d}"
            counter += 1
            vol = _to_hu(x_all[k])          
            img = nib.Nifti1Image(vol, affine=np.eye(4))
            filename = f"{vid}.nii.gz"
            nib.save(img, args.out / filename)

            manifest_rows.append({
                "volume_id": vid,
                "path": filename,
                "label": int(y_all[k]),
                "organ": "lung",
            })
            split_to_ids[our_split].append(vid)

            if (j + 1) % 100 == 0:
                print(f"  [{our_split}] wrote {j + 1}/{len(idx_pool)}")

    # Manifest CSV.
    with args.manifest.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["volume_id", "path", "label", "organ"])
        w.writeheader()
        w.writerows(manifest_rows)

    # Per-split id files (one volume_id per line).
    for split_name, ids in split_to_ids.items():
        (args.splits_dir / f"{split_name}.txt").write_text("\n".join(ids) + "\n")

    print()
    print(f"[done] manifest -> {args.manifest}  ({len(manifest_rows)} rows)")
    for split_name, ids in split_to_ids.items():
        print(f"[done] split    -> {args.splits_dir / (split_name + '.txt')}"
              f"  ({len(ids)} ids)")
    print(f"[done] volumes  -> {args.out}/  (.nii.gz, fake-HU rescaled)")


if __name__ == "__main__":
    main()
