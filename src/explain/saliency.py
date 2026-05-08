"""Localization for OmniCT.

Two attribution methods, both backbone-agnostic and applied at
inference time only:

- ``gradient_times_input`` (Simonyan et al., 2014): vanilla saliency.
  ``S(x) = |x * dy/dx|``, normalized per-volume. Useful as a sanity
  baseline; works for any model.

- ``GradCAM3D`` (Selvaraju et al., 2017): registers a forward and
  backward hook on a chosen feature layer, computes channel weights
  from the gradient of the target logit w.r.t. that layer's
  activations, and produces a (D', H', W') heatmap that is
  trilinear-upsampled to the input resolution.

We deliberately keep this small (one file, ~150 lines) because the
saliency analysis is a single qualitative figure in the report rather
than a benchmark on its own.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


# ---------------------------------------------------------------------------
# Vanilla gradient saliency
# ---------------------------------------------------------------------------


def gradient_times_input(
    model: nn.Module,
    x: torch.Tensor,
    target_class: int = 1,
    normalize: bool = True,
) -> torch.Tensor:
    """Compute |x * d_logit/d_x| as a saliency volume.

    Args:
        model: a (B, 1, D, H, W) -> (B, n_classes) classifier.
        x: input batch, shape (B, 1, D, H, W). Will be enable-grad'd.
        target_class: class id to backprop from. Default 1 = malignant.
        normalize: if True, scale each volume to [0, 1].

    Returns:
        Tensor of shape (B, 1, D, H, W) with non-negative attribution.
    """
    was_training = model.training
    model.eval()

    x = x.detach().clone().requires_grad_(True)

    logits = model(x)
    score = logits[:, target_class].sum()
    grads = torch.autograd.grad(score, x, retain_graph=False, create_graph=False)[0]

    sal = (x * grads).abs()

    if normalize:
        b = sal.shape[0]
        flat = sal.view(b, -1)
        denom = flat.amax(dim=1).clamp(min=1e-8).view(b, 1, 1, 1, 1)
        sal = sal / denom

    if was_training:
        model.train()
    return sal.detach()


# ---------------------------------------------------------------------------
# 3D Grad-CAM
# ---------------------------------------------------------------------------


class GradCAM3D:
    """Compute Grad-CAM heatmaps on a 3D model.

    Usage:
        cam = GradCAM3D(model, target_module=model.encoder.backbone.block3)
        heatmap = cam(x, target_class=1)  # (B, 1, D, H, W) in [0, 1]
        cam.close()                       # remove hooks

    `target_module` should be a module whose forward output is a
    feature map ``(B, C, D', H', W')``. For SmallCNN3D, picking
    ``model.block3`` (the deepest conv block, before global pooling) is
    a good default. For a 3D ViT, picking the last attention block's
    output (after reshape back to spatial form) works -- but plain
    gradient-times-input is usually a better fit for transformers.
    """

    def __init__(self, model: nn.Module, target_module: nn.Module) -> None:
        self.model = model
        self.target_module = target_module
        self._activation: Optional[torch.Tensor] = None
        self._gradient: Optional[torch.Tensor] = None

        self._fwd_handle = target_module.register_forward_hook(self._save_activation)
        self._bwd_handle = target_module.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _module, _inp, out):
        self._activation = out.detach()

    def _save_gradient(self, _module, _grad_in, grad_out):
        self._gradient = grad_out[0].detach()

    def __call__(
        self,
        x: torch.Tensor,
        target_class: int = 1,
        normalize: bool = True,
    ) -> torch.Tensor:
        was_training = self.model.training
        self.model.eval()

        x = x.detach().clone().requires_grad_(True)

        logits = self.model(x)
        self.model.zero_grad(set_to_none=True)
        score = logits[:, target_class].sum()
        score.backward(retain_graph=False)

        if self._activation is None or self._gradient is None:
            raise RuntimeError(
                "Grad-CAM hooks did not fire. Check that target_module is "
                "actually invoked during forward."
            )

        # weights: global-average-pool the gradients over spatial dims.
        # shape: (B, C, 1, 1, 1)
        weights = self._gradient.mean(dim=(2, 3, 4), keepdim=True)
        cam = (weights * self._activation).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(
            cam, size=x.shape[-3:], mode="trilinear", align_corners=False
        )

        if normalize:
            b = cam.shape[0]
            flat = cam.view(b, -1)
            mn = flat.amin(dim=1).view(b, 1, 1, 1, 1)
            mx = flat.amax(dim=1).clamp(min=1e-8).view(b, 1, 1, 1, 1)
            cam = (cam - mn) / (mx - mn).clamp(min=1e-8)

        if was_training:
            self.model.train()
        return cam.detach()

    def close(self) -> None:
        self._fwd_handle.remove()
        self._bwd_handle.remove()


# ---------------------------------------------------------------------------
# Plotting helper for the saliency figure
# ---------------------------------------------------------------------------


def save_saliency_panel(
    images: torch.Tensor,
    saliencies: Sequence[torch.Tensor],
    method_names: Sequence[str],
    out_path: str | Path,
    labels: Optional[Sequence[int]] = None,
    probs: Optional[Sequence[float]] = None,
    slice_axis: int = 2,
) -> None:
    """Render an (n_methods+1) x B grid PNG/PDF for the report.

    Top row is a central slice of the input volume; subsequent rows
    overlay each saliency method (hot colormap, alpha=0.5).

    Args:
        images:   (B, 1, D, H, W) input volumes.
        saliencies: list of (B, 1, D, H, W) attribution tensors, one
                    per method.
        method_names: human-readable method names for row titles.
        out_path: file to save (.pdf or .png).
        labels: optional ground-truth labels per sample (for column
                titles).
        probs: optional predicted probabilities for the positive class.
        slice_axis: which spatial axis to slice along for display.
                    0 -> sagittal, 1 -> coronal, 2 -> axial.
    """
    import matplotlib.pyplot as plt  # local import keeps this file light

    images = images.detach().cpu()
    saliencies = [s.detach().cpu() for s in saliencies]
    b = images.shape[0]
    n_rows = len(saliencies) + 1

    fig, axes = plt.subplots(n_rows, b, figsize=(2.6 * b, 2.6 * n_rows), squeeze=False)

    def _mid_slice(vol):
        idx = vol.shape[slice_axis + 2] // 2  # +2 because (B, C, D, H, W)
        return vol.select(dim=slice_axis + 2, index=idx).squeeze(1)

    img_mid = _mid_slice(images)

    for j in range(b):
        ax = axes[0][j]
        ax.imshow(img_mid[j].numpy(), cmap="gray")
        ax.set_xticks([]); ax.set_yticks([])
        title = []
        if labels is not None:
            title.append(f"y={labels[j]}")
        if probs is not None:
            title.append(f"p={probs[j]:.2f}")
        if title:
            ax.set_title(" ".join(title), fontsize=9)
        if j == 0:
            ax.set_ylabel("input", fontsize=9)

    for i, (sal, name) in enumerate(zip(saliencies, method_names), start=1):
        sal_mid = _mid_slice(sal)
        for j in range(b):
            ax = axes[i][j]
            ax.imshow(img_mid[j].numpy(), cmap="gray")
            ax.imshow(sal_mid[j].numpy(), cmap="hot", alpha=0.5)
            ax.set_xticks([]); ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(name, fontsize=9)

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
