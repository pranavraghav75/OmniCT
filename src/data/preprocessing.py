"""3D CT preprocessing pipeline.

1. Load NIfTI as a (1, D, H, W) tensor with metadata.
2. Hounsfield-Unit clip to a soft-tissue window (default [-1000, 400] HU)
   and min-max normalize to [0, 1].
3. Resize (deterministic trilinear) to a fixed spatial size for batching.
4. Train transforms add light, anatomy-respecting augmentation (random
   flip along left-right, random 90 degree rotation in the axial plane,
   small intensity jitter).

Notes
-----
We deliberately omit ``Spacingd``, ``Orientationd``, and
``CropForegroundd`` from this pipeline. These transforms call legacy
NumPy idioms (e.g., ``ndarray.ptp``) that NumPy 2.x removed, breaking
``Spacingd`` on Colab even with MONAI 1.5+. Avoiding them entirely is
both more numerically transparent and immune to that fragility.

If you bring in raw CT scans whose voxel spacing is *not* already
isotropic, pre-resample them on disk (e.g., with SimpleITK or
``monai.transforms.Spacing`` in a one-off offline script) before they
reach this pipeline. The ``spacing`` argument is kept for backward
compatibility but is currently unused.
"""

from __future__ import annotations
from typing import Sequence
from monai import transforms as mt


def build_train_transforms(
    spacing: Sequence[float] = (1.0, 1.0, 1.0),  # kept for backward compat
    hu_window: Sequence[float] = (-1000.0, 400.0),
    spatial_size: Sequence[int] = (32, 32, 32),
    keys: Sequence[str] = ("image",),
) -> mt.Compose:
    del spacing  # unused; resampling is expected to happen offline
    a, b = float(hu_window[0]), float(hu_window[1])
    return mt.Compose(
        [
            mt.LoadImaged(keys=keys, image_only=True),
            mt.EnsureChannelFirstd(keys=keys),
            mt.ScaleIntensityRanged(
                keys=keys, a_min=a, a_max=b, b_min=0.0, b_max=1.0, clip=True
            ),
            mt.Resized(keys=keys, spatial_size=tuple(spatial_size), mode="trilinear"),
            mt.RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
            mt.RandRotate90d(keys=keys, prob=0.25, spatial_axes=(1, 2)),
            mt.RandShiftIntensityd(keys=keys, offsets=0.05, prob=0.5),
            mt.EnsureTyped(keys=keys),
        ]
    )


def build_eval_transforms(
    spacing: Sequence[float] = (1.0, 1.0, 1.0),  # kept for backward compat
    hu_window: Sequence[float] = (-1000.0, 400.0),
    spatial_size: Sequence[int] = (32, 32, 32),
    keys: Sequence[str] = ("image",),
) -> mt.Compose:
    del spacing
    a, b = float(hu_window[0]), float(hu_window[1])
    return mt.Compose(
        [
            mt.LoadImaged(keys=keys, image_only=True),
            mt.EnsureChannelFirstd(keys=keys),
            mt.ScaleIntensityRanged(
                keys=keys, a_min=a, a_max=b, b_min=0.0, b_max=1.0, clip=True
            ),
            mt.Resized(keys=keys, spatial_size=tuple(spatial_size), mode="trilinear"),
            mt.EnsureTyped(keys=keys),
        ]
    )