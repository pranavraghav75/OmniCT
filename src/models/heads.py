"""Classification heads that sit on top of a (possibly frozen) feature
extractor.

- `LinearProbeHead`: a single linear layer (Pai et al. 2024 style).
- `MLPHead`: a small 2-layer MLP for slightly more capacity.
- `MultiOrganHead`: per-organ heads sharing a common trunk, used for our
  achievable-plan multi-organ setup.
"""

from __future__ import annotations

import torch
from torch import nn


class LinearProbeHead(nn.Module):
    def __init__(self, feature_dim: int, n_classes: int = 2) -> None:
        super().__init__()
        self.fc = nn.Linear(feature_dim, n_classes)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.fc(feats)


class MLPHead(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        n_classes: int = 2,
        hidden: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.net(feats)


class MultiOrganHead(nn.Module):
    """One classification head per organ, sharing the same backbone features.

    Forward expects (feats, organ_id). At training time, only the head
    matching the organ for each sample contributes to the loss.
    """

    def __init__(
        self,
        feature_dim: int,
        n_organs: int,
        n_classes: int = 2,
        hidden: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.heads = nn.ModuleList(
            [MLPHead(feature_dim, n_classes, hidden, dropout) for _ in range(n_organs)]
        )

    def forward(self, feats: torch.Tensor, organ_ids: torch.Tensor) -> torch.Tensor:
        b = feats.shape[0]
        n_classes = self.heads[0].net[-1].out_features
        out = feats.new_zeros((b, n_classes))
        for organ_id, head in enumerate(self.heads):
            mask = organ_ids == organ_id
            if mask.any():
                out[mask] = head(feats[mask])
        return out
