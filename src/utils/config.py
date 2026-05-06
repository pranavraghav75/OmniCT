"""YAML config loader with merge-on-top semantics.

Configs use OmegaConf so we get dotted access (`cfg.train.lr`) and
CLI override support (`key=value` style).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from omegaconf import DictConfig, OmegaConf


def load_config(
    path: str | Path,
    overrides: Iterable[str] | None = None,
) -> DictConfig:
    """Load a YAML config from `path` and apply CLI-style overrides.

    `overrides` is an iterable of `dotted.key=value` strings, e.g.
    `["train.lr=1e-4", "seed=7"]`.
    """
    base = OmegaConf.load(str(path))
    if overrides:
        cli = OmegaConf.from_dotlist(list(overrides))
        base = OmegaConf.merge(base, cli)
    assert isinstance(base, DictConfig)
    return base


def save_config(cfg: DictConfig, path: str | Path) -> None:
    """Persist the resolved config next to a run's outputs."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, str(path))


def to_container(cfg: DictConfig) -> dict[str, Any]:
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
