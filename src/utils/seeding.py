"""Deterministic seeding for reproducibility.

Sets seeds for python `random`, numpy, and torch (CPU + CUDA), and
toggles cudnn determinism. Call `set_seed` once at the top of every
entry-point script (training, evaluation, data download, etc.).
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed all relevant RNGs.

    Args:
        seed: integer seed.
        deterministic: if True, force cudnn to deterministic mode. This may
            slow training but is required for bit-exact reproducibility.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
