"""OmniCT data package.

`preprocessing` requires MONAI (heavy dependency). It's imported lazily
so that synthetic / CI runs that don't touch real NIfTI files can avoid
the import cost.
"""

from .datasets import OmniCTDataset, SyntheticCTDataset


def build_train_transforms(*args, **kwargs):
    from .preprocessing import build_train_transforms as _impl

    return _impl(*args, **kwargs)


def build_eval_transforms(*args, **kwargs):
    from .preprocessing import build_eval_transforms as _impl

    return _impl(*args, **kwargs)


__all__ = [
    "build_train_transforms",
    "build_eval_transforms",
    "OmniCTDataset",
    "SyntheticCTDataset",
]
