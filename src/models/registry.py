"""Builds a model from a config dict.

Supported `model.kind` values:
    - "prior"         : ClassPriorClassifier (no training).
    - "cnn3d"         : SmallCNN3D, trained from scratch.
    - "linear_probe"  : frozen FoundationEncoder + LinearProbeHead.
    - "lora"          : FoundationEncoder with injected LoRA + MLPHead.
"""

from __future__ import annotations

import torch
from torch import nn

from .baselines import ClassPriorClassifier, SmallCNN3D
from .foundation import build_foundation_encoder
from .heads import LinearProbeHead, MLPHead
from .lora import inject_lora


# Keys in `cfg.model` that should be forwarded to the foundation-model
# loader as keyword arguments. This is how the YAML can pass
# backbone-specific options (3DINO `repo_path`, SPECTRE `patch_size`, …)
# without us hard-wiring each one in the registry.
_BACKBONE_KWARG_KEYS = (
    "repo_path",
    "config_name",
    "expected_size",
    "model_name",
    "patch_size",
    "grid_size",
)


class _EncoderHead(nn.Module):
    def __init__(self, encoder: nn.Module, head: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)
        return self.head(feats)


def _backbone_kwargs(cfg) -> dict:
    out: dict[str, object] = {}
    for k in _BACKBONE_KWARG_KEYS:
        v = cfg.model.get(k)
        if v is None:
            continue
        if k in ("expected_size", "patch_size", "grid_size"):
            v = tuple(int(x) for x in v)
        out[k] = v
    return out


def build_model(cfg) -> nn.Module:
    kind = cfg.model.kind

    if kind == "prior":
        return ClassPriorClassifier(n_classes=int(cfg.model.n_classes))

    if kind == "cnn3d":
        return SmallCNN3D(
            in_channels=int(cfg.data.in_channels),
            n_classes=int(cfg.model.n_classes),
            widths=tuple(cfg.model.widths),
            dropout=float(cfg.model.dropout),
        )

    if kind == "linear_probe":
        encoder = build_foundation_encoder(
            name=str(cfg.model.backbone),
            checkpoint_path=str(cfg.model.checkpoint_path)
            if cfg.model.get("checkpoint_path")
            else None,
            feature_dim=int(cfg.model.feature_dim),
            **_backbone_kwargs(cfg),
        )
        head = LinearProbeHead(
            feature_dim=encoder.feature_dim,
            n_classes=int(cfg.model.n_classes),
        )
        return _EncoderHead(encoder, head)

    if kind == "lora":
        encoder = build_foundation_encoder(
            name=str(cfg.model.backbone),
            checkpoint_path=str(cfg.model.checkpoint_path)
            if cfg.model.get("checkpoint_path")
            else None,
            feature_dim=int(cfg.model.feature_dim),
            **_backbone_kwargs(cfg),
        )
        encoder.frozen = False

        n_lora = inject_lora(
            encoder,
            target_substrings=tuple(cfg.model.lora.target_substrings),
            r=int(cfg.model.lora.r),
            alpha=int(cfg.model.lora.alpha),
            dropout=float(cfg.model.lora.dropout),
        )
        for name, p in encoder.named_parameters():
            if name.endswith(".A") or name.endswith(".B"):
                p.requires_grad_(True)
            else:
                p.requires_grad_(False)

        head = MLPHead(
            feature_dim=encoder.feature_dim,
            n_classes=int(cfg.model.n_classes),
            hidden=int(cfg.model.head_hidden),
            dropout=float(cfg.model.head_dropout),
        )
        model = _EncoderHead(encoder, head)
        model._n_lora_layers = n_lora  # type: ignore[attr-defined]
        return model

    raise ValueError(f"Unknown model.kind: {kind!r}")
