"""Wrappers around 3D medical foundation-model encoders.

We support three backbones via a uniform `FoundationEncoder` interface
that always takes a tensor of shape ``(B, 1, D, H, W)`` with values in
``[0, 1]`` (i.e. the output of `src.data.preprocessing`) and returns a
``(B, feature_dim)`` embedding.

Backbones:

- ``random_init``: a `SmallCNN3D` initialized from scratch. Used for CI /
  smoke tests / development on machines without the foundation-model
  weights downloaded.

- ``3dino``: the 3DINO-ViT-Large model from Xu et al., 2025
  (npj Digital Medicine). Requires a local clone of
  https://github.com/AICONSlab/3DINO and the teacher checkpoint
  downloaded from https://huggingface.co/AICONSlab/3DINO-ViT.

- ``spectre``: SPECTRE (Claessens et al., 2026) — a 3D ViT trained with
  SSL + cross-modal vision-language alignment. Installed via
  ``pip install spectre-fm``.

The adapter classes (`_DINO3DAdapter`, `_SpectreAdapter`) handle the
quirks of each backbone (input normalization, crop / patch grid
construction, output pooling) so the rest of the codebase stays simple.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch import nn

from .baselines import SmallCNN3D


# ---------------------------------------------------------------------------
# Generic encoder wrapper
# ---------------------------------------------------------------------------


class FoundationEncoder(nn.Module):
    """Wraps a backbone so callers always get a ``(B, feature_dim)`` tensor.

    The backbone may itself output a ``(B, C, D, H, W)`` feature map; this
    class globally pools to a single vector if so.
    """

    def __init__(self, backbone: nn.Module, feature_dim: int, freeze: bool = True) -> None:
        super().__init__()
        self.backbone = backbone
        self.feature_dim = feature_dim
        self.frozen = freeze
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

    def train(self, mode: bool = True):  # type: ignore[override]
        super().train(mode)
        if self.frozen:
            self.backbone.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ctx = torch.no_grad() if self.frozen else torch.enable_grad()
        with ctx:
            if hasattr(self.backbone, "extract_features"):
                out = self.backbone.extract_features(x)
            elif hasattr(self.backbone, "features"):
                out = self.backbone.features(x)
            else:
                out = self.backbone(x)
        if out.ndim > 2:
            # Global average pool spatial dims down to (B, C).
            out = out.mean(dim=tuple(range(2, out.ndim)))
        return out


# ---------------------------------------------------------------------------
# Random-init stub (no real pretraining)
# ---------------------------------------------------------------------------


def _load_random_init(feature_dim: int = 128, **_: object) -> FoundationEncoder:
    backbone = SmallCNN3D(n_classes=feature_dim)
    return FoundationEncoder(backbone=backbone, feature_dim=backbone.feature_dim, freeze=True)


# ---------------------------------------------------------------------------
# 3DINO (Xu et al., 2025)
# ---------------------------------------------------------------------------


class _DINO3DAdapter(nn.Module):
    """Adapts the 3DINO-ViT model so it consumes ``(B, 1, D, H, W)`` in
    ``[0, 1]`` and emits ``(B, feature_dim)``.

    3DINO expects inputs normalized to ``[-1, 1]``. We linearly map our
    pipeline's ``[0, 1]`` to that range. If the spatial size differs from
    the size 3DINO was trained at, we trilinear-interpolate to match.
    """

    def __init__(
        self,
        model: nn.Module,
        feature_dim: int = 1024,
        expected_size: tuple[int, int, int] = (112, 112, 112),
    ) -> None:
        super().__init__()
        self.model = model
        self.feature_dim = feature_dim
        self.expected_size = tuple(expected_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-3:] != self.expected_size:
            x = F.interpolate(
                x, size=self.expected_size, mode="trilinear", align_corners=False
            )
        x = x * 2.0 - 1.0  # [0, 1] -> [-1, 1]
        return self.model(x)


def _load_3dino(
    checkpoint_path: str,
    repo_path: Optional[str] = None,
    config_name: str = "train/vit3d_highres",
    feature_dim: int = 1024,
    expected_size: tuple[int, int, int] = (112, 112, 112),
    **_: object,
) -> FoundationEncoder:
    """Load 3DINO-ViT pretrained weights.

    Args:
        checkpoint_path: path to ``teacher_checkpoint.pth`` (downloaded
            from the HuggingFace mirror or saved during pretraining).
        repo_path: path to a local clone of
            https://github.com/AICONSlab/3DINO. Required because 3DINO
            ships its model code as a regular Python package, not as a
            pip-installable wheel. If ``None``, we fall back to checking
            the ``OMNICT_3DINO_REPO`` environment variable.
        config_name: 3DINO config string passed to
            ``dinov2.configs.load_and_merge_config_3d``. The default
            ``'train/vit3d_highres'`` matches the published ViT-Large
            high-res adaptation.
        feature_dim: output dim of the CLS token (1024 for ViT-Large).
        expected_size: input spatial size. 3DINO uses 112^3 at high-res
            adaptation; we resize incoming volumes to this size.
    """
    import os
    import sys

    if repo_path is None:
        repo_path = os.environ.get("OMNICT_3DINO_REPO")
    if not repo_path:
        raise ValueError(
            "3DINO requires repo_path (clone https://github.com/AICONSlab/3DINO "
            "and pass its path, or set the OMNICT_3DINO_REPO env var)."
        )
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)

    try:
        from dinov2.configs import load_and_merge_config_3d  # type: ignore
        from dinov2.eval.setup import build_model_for_eval  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "Failed to import 3DINO modules from "
            f"repo_path={repo_path!r}. Make sure the path points to a "
            "valid clone of https://github.com/AICONSlab/3DINO. "
            f"Original error: {e}"
        ) from e

    cfg = load_and_merge_config_3d(config_name)
    # `build_model_for_eval` handles checkpoint loading, including the
    # quirk where 3DINO saves the teacher under a `teacher` key.
    model = build_model_for_eval(cfg, checkpoint_path)
    model.eval()

    adapter = _DINO3DAdapter(model, feature_dim=feature_dim, expected_size=expected_size)
    return FoundationEncoder(backbone=adapter, feature_dim=feature_dim, freeze=True)


# ---------------------------------------------------------------------------
# SPECTRE (Claessens et al., 2026)
# ---------------------------------------------------------------------------


class _SpectreAdapter(nn.Module):
    """Adapts SPECTRE to take ``(B, 1, D, H, W)`` in ``[0, 1]``.

    SPECTRE's ``SpectreImageFeatureExtractor`` consumes a *crop grid* —
    a tensor of shape ``(B, num_crops, C, pH, pW, pD)`` — together with
    a ``grid_size`` argument. We:

    1. Trilinear-interpolate the input to a size divisible by the patch
       size (defaults to ``(128, 128, 64)``, the published patch
       resolution).
    2. View / permute / reshape into the crop grid layout.
    3. Forward through SPECTRE.
    4. Pool spatial / crop dims down to a single ``(B, feature_dim)``
       vector.

    Note: SPECTRE was published as ``(H, W, D)`` ordering. Our pipeline
    delivers ``(D, H, W)`` per MONAI's ``EnsureChannelFirstd``+
    ``Spacingd`` convention. The reshape works either way as long as it
    is internally consistent — we treat the trailing 3 dims uniformly.
    """

    def __init__(
        self,
        model: nn.Module,
        feature_dim: int = 1024,
        patch_size: tuple[int, int, int] = (128, 128, 64),
        grid_size: tuple[int, int, int] = (3, 3, 4),
    ) -> None:
        super().__init__()
        self.model = model
        self.feature_dim = feature_dim
        self.patch_size = tuple(patch_size)
        self.grid_size = tuple(grid_size)

    @property
    def expected_size(self) -> tuple[int, int, int]:
        gh, gw, gd = self.grid_size
        ph, pw, pd = self.patch_size
        return (gh * ph, gw * pw, gd * pd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c = x.shape[0], x.shape[1]
        if x.shape[-3:] != self.expected_size:
            x = F.interpolate(
                x, size=self.expected_size, mode="trilinear", align_corners=False
            )

        gh, gw, gd = self.grid_size
        ph, pw, pd = self.patch_size
        # (B, C, gh*ph, gw*pw, gd*pd) -> (B, num_crops, C, ph, pw, pd)
        x = (
            x.view(b, c, gh, ph, gw, pw, gd, pd)
            .permute(0, 2, 4, 6, 1, 3, 5, 7)
            .reshape(b, gh * gw * gd, c, ph, pw, pd)
        )

        feats = self.model(x, grid_size=self.grid_size)

        # SPECTRE's combiner output can be (B, F) or higher-rank; pool
        # any extra dims away.
        if feats.ndim > 2:
            feats = feats.flatten(2).mean(dim=-1)
        return feats


def _load_spectre(
    checkpoint_path: Optional[str] = None,
    model_name: str = "spectre-large-pretrained",
    patch_size: tuple[int, int, int] = (128, 128, 64),
    grid_size: tuple[int, int, int] = (3, 3, 4),
    feature_dim: int = 1024,
    **_: object,
) -> FoundationEncoder:
    """Load SPECTRE pretrained weights via the ``spectre-fm`` package.

    Args:
        checkpoint_path: optional path to a custom ``state_dict``. If
            ``None``, weights bundled with ``MODEL_CONFIGS[model_name]``
            are used (the package ships pretrained weights via HF Hub).
        model_name: key into ``spectre.MODEL_CONFIGS``. Default
            ``'spectre-large-pretrained'`` matches the README example.
        patch_size: spatial size of each crop (H, W, D). Must match
            what the model was trained on.
        grid_size: number of crops per axis. Total volume size is
            ``grid_size * patch_size``.
        feature_dim: output dim of the SPECTRE feature extractor.
            ViT-Large -> 1024.
    """
    try:
        from spectre import MODEL_CONFIGS, SpectreImageFeatureExtractor  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "Failed to import 'spectre'. Install with `pip install spectre-fm`. "
            f"Original error: {e}"
        ) from e

    if model_name not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown SPECTRE model_name {model_name!r}. "
            f"Available: {sorted(MODEL_CONFIGS)}"
        )

    config = MODEL_CONFIGS[model_name]
    model = SpectreImageFeatureExtractor.from_config(config)
    model.eval()

    if checkpoint_path is not None:
        sd = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(sd, dict):
            for key in ("state_dict", "model", "model_state_dict"):
                if key in sd and isinstance(sd[key], dict):
                    sd = sd[key]
                    break
        # Strip a possible 'module.' prefix from DDP-saved checkpoints.
        sd = {k.removeprefix("module."): v for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            print(
                f"[spectre] loaded with strict=False. "
                f"missing={len(missing)}, unexpected={len(unexpected)}"
            )

    adapter = _SpectreAdapter(
        model,
        feature_dim=feature_dim,
        patch_size=patch_size,
        grid_size=grid_size,
    )
    return FoundationEncoder(backbone=adapter, feature_dim=feature_dim, freeze=True)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


BACKBONES: dict[str, Callable[..., FoundationEncoder]] = {
    "random_init": _load_random_init,
    "3dino": _load_3dino,
    "spectre": _load_spectre,
}


def build_foundation_encoder(
    name: str,
    checkpoint_path: Optional[str] = None,
    feature_dim: int = 128,
    **kwargs: object,
) -> FoundationEncoder:
    """Build a `FoundationEncoder` by name.

    Extra keyword arguments are forwarded to the per-backbone loader
    (e.g. ``repo_path`` for 3DINO, ``patch_size``/``grid_size`` for
    SPECTRE). This lets `src.models.registry.build_model` plumb
    backbone-specific config through the YAML.
    """
    if name not in BACKBONES:
        raise ValueError(f"Unknown backbone {name!r}. Available: {sorted(BACKBONES)}")
    if name == "random_init":
        return BACKBONES[name](feature_dim=feature_dim, **kwargs)
    if checkpoint_path is None and name == "3dino":
        raise ValueError(
            "3dino requires a checkpoint_path (path to teacher_checkpoint.pth)."
        )
    return BACKBONES[name](
        checkpoint_path=checkpoint_path,
        feature_dim=feature_dim,
        **kwargs,
    )
