"""Baseline models for OmniCT.

- `ClassPriorClassifier`: returns the empirical class prior; serves as the
  trivial lower bound a real model must beat.
- `SmallCNN3D`: a compact 3D ConvNet trained from scratch, intended as a
  no-pretraining baseline.
"""

from __future__ import annotations
import torch
from torch import nn

class ClassPriorClassifier(nn.Module):
    """Predicts a fixed class-prior probability for every input.

    Fit by calling `.fit_prior(labels)` on the training labels once.
    """

    def __init__(self, n_classes: int = 2) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.register_buffer("prior", torch.full((n_classes,), 1.0 / n_classes))

    def fit_prior(self, labels: torch.Tensor) -> None:
        counts = torch.bincount(labels.long(), minlength=self.n_classes).float()
        self.prior = counts / counts.sum().clamp(min=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        logits = torch.log(self.prior.clamp(min=1e-8)).unsqueeze(0).expand(b, -1)
        return logits.to(x.device)


def _conv_block(in_ch: int, out_ch: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm3d(out_ch),
        nn.ReLU(inplace=True),
    )

class SmallCNN3D(nn.Module):
    """A modest 3D CNN — 4 conv blocks with strided downsampling + global pool.

    Designed to fit comfortably on a single 16 GB GPU at 96^3 input size.
    """

    def __init__(
        self,
        in_channels: int = 1,
        n_classes: int = 2,
        widths: tuple[int, int, int, int] = (16, 32, 64, 128),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        c1, c2, c3, c4 = widths
        self.stem = _conv_block(in_channels, c1)
        self.block1 = nn.Sequential(_conv_block(c1, c2, stride=2), _conv_block(c2, c2))
        self.block2 = nn.Sequential(_conv_block(c2, c3, stride=2), _conv_block(c3, c3))
        self.block3 = nn.Sequential(_conv_block(c3, c4, stride=2), _conv_block(c4, c4))
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(c4, n_classes)
        self.feature_dim = c4

    def features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x).flatten(1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.features(x)
        return self.fc(self.dropout(feats))