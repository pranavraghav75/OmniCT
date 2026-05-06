from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class OmniCTDataset(Dataset):
    """3D CT volume + binary malignancy label, optionally with organ id.

    Manifest CSV must contain columns:
        - volume_id : str, unique identifier
        - path      : str, NIfTI path (absolute or relative to `data_root`)
        - label     : int, 0 = benign, 1 = malignant
        - organ     : str (optional), used for per-organ analysis

    Args:
        manifest: path to CSV.
        data_root: directory to resolve relative paths against.
        split_ids: optional iterable of `volume_id`s to keep (for train/val/test).
        transform: a MONAI Compose (or any callable on a dict).
        organ_to_id: dict mapping organ string -> int. If None, organs are ignored.
    """

    def __init__(
        self,
        manifest: str | Path,
        data_root: Optional[str | Path] = None,
        split_ids: Optional[list[str]] = None,
        transform: Optional[Callable] = None,
        organ_to_id: Optional[dict[str, int]] = None,
    ) -> None:
        df = pd.read_csv(manifest)
        if split_ids is not None:
            df = df[df["volume_id"].isin(set(split_ids))].reset_index(drop=True)

        required = {"volume_id", "path", "label"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Manifest is missing required columns: {missing}")

        self.df = df
        self.data_root = Path(data_root) if data_root is not None else None
        self.transform = transform
        self.organ_to_id = organ_to_id

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        path = Path(row["path"])
        if self.data_root is not None and not path.is_absolute():
            path = self.data_root / path

        sample = {
            "image": str(path),
            "label": int(row["label"]),
            "volume_id": str(row["volume_id"]),
        }
        if "organ" in row and self.organ_to_id is not None:
            sample["organ"] = int(self.organ_to_id.get(str(row["organ"]), -1))

        if self.transform is not None:
            sample = self.transform(sample)

        return sample


class SyntheticCTDataset(Dataset):
    """Random 3D volumes for smoke tests.

    Useful so the training loop can be exercised end-to-end on a laptop
    without needing the real (very large) CT data.
    """

    def __init__(
        self,
        n_samples: int = 32,
        spatial_size: tuple[int, int, int] = (96, 96, 96),
        n_classes: int = 2,
        seed: int = 0,
    ) -> None:
        self.n_samples = n_samples
        self.spatial_size = spatial_size
        self.n_classes = n_classes
        rng = np.random.default_rng(seed)
        self.labels = rng.integers(0, n_classes, size=n_samples).tolist()
        self._seed = seed

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict:
        g = torch.Generator().manual_seed(self._seed * 100003 + idx)
        x = torch.randn((1, *self.spatial_size), generator=g)
        x = x + 0.5 * float(self.labels[idx])
        return {
            "image": x,
            "label": int(self.labels[idx]),
            "volume_id": f"synthetic_{idx:06d}",
        }
