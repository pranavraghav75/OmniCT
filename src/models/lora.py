"""Tiny self-contained LoRA layer + helper to inject LoRA into a torch
module. We do not depend on peft here so this code remains transparent
and works for arbitrary 3D backbones.

LoRA(W) y = W x + (B A) x, with A ∈ R^{r x in}, B ∈ R^{out xr}, B initialized
to 0 so the adapted layer starts as the identity (Hu et al., 2021).
"""

from __future__ import annotations
import math
from typing import Iterable
import torch
from torch import nn

class LoRALinear(nn.Module):
    def __init__(
        self,
        base: nn.Linear,
        r: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if r <= 0:
            raise ValueError("LoRA rank must be > 0")
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        in_f = base.in_features
        out_f = base.out_features
        self.r = r
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.A = nn.Parameter(torch.empty(r, in_f))
        self.B = nn.Parameter(torch.zeros(out_f, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        delta = self.dropout(x) @ self.A.T @ self.B.T
        return out + self.scaling * delta


def inject_lora(
    module: nn.Module,
    target_substrings: Iterable[str] = ("attn", "mlp", "qkv", "proj"),
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.0,
) -> int:
    """Replace every `nn.Linear` whose qualified name contains any of
    `target_substrings` with a `LoRALinear` wrapping it.

    Returns the number of layers wrapped. This intentionally mutates the
    module in place so the caller can keep a single reference.
    """
    n_replaced = 0
    for name, child in list(module.named_modules()):
        if not isinstance(child, nn.Linear):
            continue
        if not any(s in name for s in target_substrings):
            continue
        parent_name, _, attr = name.rpartition(".")
        parent = module.get_submodule(parent_name) if parent_name else module
        wrapped = LoRALinear(child, r=r, alpha=alpha, dropout=dropout)
        setattr(parent, attr, wrapped)
        n_replaced += 1
    return n_replaced


def trainable_parameters(module: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return trainable, total
