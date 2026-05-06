"""Lightweight logger factory used across training scripts."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


_DEFAULT_FMT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(
    name: str = "omnict",
    level: int = logging.INFO,
    log_file: str | Path | None = None,
) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    fmt = logging.Formatter(_DEFAULT_FMT)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    logger.propagate = False
    return logger
