"""OmniCT data package.

`preprocessing` requires MONAI (heavy dependency). It's imported lazily
so that synthetic / CI runs that don't touch real NIfTI files can avoid
the import cost.
"""

from .datasets import OmniCTDataset, SyntheticCTDataset


def _patch_monai_max_seed() -> None:
    """MONAI < 1.4 sets ``MAX_SEED = 2**32`` in
    ``monai.utils.misc``/``monai.transforms.{compose,transform}``. On
    NumPy 2.x, evaluating ``uint32_scalar % 2**32`` raises
    ``OverflowError: Python integer 4294967296 out of bounds for uint32``
    because ``2**32`` itself does not fit in ``uint32`` (max is
    ``2**32 - 1``). Lowering ``MAX_SEED`` by 1 in every module that
    holds a copy keeps every downstream call (``randint(MAX_SEED, ...)``,
    ``_seed % MAX_SEED``) inside the uint32 range.

    Safe to call repeatedly. Called automatically the first time MONAI
    transforms are imported through this package.
    """
    try:
        import numpy as np

        SAFE_MAX = int(np.iinfo(np.uint32).max)  # 2**32 - 1
        mod_names = (
            "monai.utils.misc",
            "monai.transforms.transform",
            "monai.transforms.compose",
        )
        for name in mod_names:
            try:
                mod = __import__(name, fromlist=["MAX_SEED"])
                if getattr(mod, "MAX_SEED", None) and mod.MAX_SEED > SAFE_MAX:
                    mod.MAX_SEED = SAFE_MAX
            except Exception:
                continue
    except Exception:
        pass


def build_train_transforms(*args, **kwargs):
    _patch_monai_max_seed()
    from .preprocessing import build_train_transforms as _impl

    return _impl(*args, **kwargs)


def build_eval_transforms(*args, **kwargs):
    _patch_monai_max_seed()
    from .preprocessing import build_eval_transforms as _impl

    return _impl(*args, **kwargs)


__all__ = [
    "build_train_transforms",
    "build_eval_transforms",
    "OmniCTDataset",
    "SyntheticCTDataset",
]
