"""3D CT preprocessing pipeline.

1. Load NIfTI as a (1, D, H, W) tensor with metadata.
2. Reorient to RAS so all volumes share an anatomical convention.
3. Resample to an isotropic voxel spacing (default 1.5 mm). This is what
   lets a 3D ViT's positional encoding be meaningful across scans.
4. Hounsfield-Unit clip to a soft-tissue window (default [-1000, 400] HU)
   to suppress bone / air noise and emphasize lesion-relevant tissue.
5. Min-max normalize to [0, 1].
6. Crop / pad to a fixed spatial size for batching.

Train transforms add light, anatomy-respecting augmentation
(random flip along left-right, small rotations, small intensity jitter).
"""

from __future__ import annotations
from typing import Sequence
from monai import transforms as mt

def build_train_transforms(
    spacing: Sequence[float] = (1.5, 1.5, 1.5),
    hu_window: Sequence[float] = (-1000.0, 400.0),
    spatial_size: Sequence[int] = (96, 96, 96),
    keys: Sequence[str] = ("image",),
) -> mt.Compose:
    a, b = float(hu_window[0]), float(hu_window[1])
    return mt.Compose(
        [
            mt.LoadImaged(keys=keys, image_only=True),
            mt.EnsureChannelFirstd(keys=keys),
            mt.Orientationd(keys=keys, axcodes="RAS"),
            mt.Spacingd(keys=keys, pixdim=tuple(spacing), mode="bilinear"),
            mt.ScaleIntensityRanged(
                keys=keys, a_min=a, a_max=b, b_min=0.0, b_max=1.0, clip=True
            ),
            mt.CropForegroundd(keys=keys, source_key=keys[0], allow_smaller=True),
            mt.SpatialPadd(keys=keys, spatial_size=tuple(spatial_size)),
            mt.RandSpatialCropd(keys=keys, roi_size=tuple(spatial_size), random_size=False),
            mt.RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
            mt.RandRotate90d(keys=keys, prob=0.25, spatial_axes=(1, 2)),
            mt.RandShiftIntensityd(keys=keys, offsets=0.05, prob=0.5),
            mt.EnsureTyped(keys=keys),
        ]
    )

def build_eval_transforms(
    spacing: Sequence[float] = (1.5, 1.5, 1.5),
    hu_window: Sequence[float] = (-1000.0, 400.0),
    spatial_size: Sequence[int] = (96, 96, 96),
    keys: Sequence[str] = ("image",),
) -> mt.Compose:
    a, b = float(hu_window[0]), float(hu_window[1])
    return mt.Compose(
        [
            mt.LoadImaged(keys=keys, image_only=True),
            mt.EnsureChannelFirstd(keys=keys),
            mt.Orientationd(keys=keys, axcodes="RAS"),
            mt.Spacingd(keys=keys, pixdim=tuple(spacing), mode="bilinear"),
            mt.ScaleIntensityRanged(
                keys=keys, a_min=a, a_max=b, b_min=0.0, b_max=1.0, clip=True
            ),
            mt.CropForegroundd(keys=keys, source_key=keys[0], allow_smaller=True),
            mt.SpatialPadd(keys=keys, spatial_size=tuple(spatial_size)),
            mt.CenterSpatialCropd(keys=keys, roi_size=tuple(spatial_size)),
            mt.EnsureTyped(keys=keys),
        ]
    )