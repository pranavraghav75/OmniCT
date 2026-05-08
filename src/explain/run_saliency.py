"""Compute saliency / Grad-CAM panels for a trained checkpoint.

Run:
    python -m src.explain.run_saliency \
        --config src/configs/lora.yaml \
        --checkpoint results/lora_seed0/best.pt \
        --n_samples 6 \
        --out report/figures/saliency_panel.pdf

This script:
    1. Reconstructs the same model the training run used (from the YAML).
    2. Loads the checkpoint.
    3. Pulls a few validation samples (synthetic or real, per the config).
    4. Computes gradient-times-input + 3D Grad-CAM heatmaps.
    5. Saves a single PDF/PNG figure ready for the Analysis section.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data import (
    OmniCTDataset,
    SyntheticCTDataset,
    build_eval_transforms,
)
from src.explain.saliency import (
    GradCAM3D,
    gradient_times_input,
    save_saliency_panel,
)
from src.models import build_model
from src.utils import load_config, set_seed


def _build_eval_loader(cfg, n_samples: int) -> DataLoader:
    if cfg.data.synthetic:
        ds = SyntheticCTDataset(
            n_samples=n_samples,
            spatial_size=tuple(cfg.data.spatial_size),
            n_classes=int(cfg.model.n_classes),
            seed=int(cfg.seed),
        )
    else:
        eval_tf = build_eval_transforms(
            spacing=tuple(cfg.data.spacing),
            hu_window=tuple(cfg.data.hu_window),
            spatial_size=tuple(cfg.data.spatial_size),
        )
        ids_path = Path(cfg.data.val_ids)
        split_ids = ids_path.read_text().strip().splitlines() if ids_path.exists() else None
        ds = OmniCTDataset(
            manifest=cfg.data.manifest,
            data_root=cfg.data.data_root,
            split_ids=split_ids,
            transform=eval_tf,
        )
        # Subsample down to n_samples for the figure.
        if len(ds) > n_samples:
            indices = list(range(0, len(ds), max(1, len(ds) // n_samples)))[:n_samples]
            ds = torch.utils.data.Subset(ds, indices)
    return DataLoader(ds, batch_size=n_samples, shuffle=False, num_workers=0)


def _resolve_target_module(model) -> torch.nn.Module:
    """Pick a sensible default Grad-CAM target depending on model kind.

    For SmallCNN3D-based models, the deepest conv block is best.
    For an `_EncoderHead(encoder, head)` (linear_probe / lora), we
    descend into ``encoder.backbone`` and try to find a ``block3``
    attribute (i.e. the SmallCNN3D fallback). Real ViT backbones don't
    expose a clean spatial feature map, so we default to using only
    gradient-times-input in that case.
    """
    candidates = ("block3", "block2", "stem")
    for attr_chain in (
        ("encoder", "backbone"),
        ("encoder", "backbone", "model"),
        ("backbone",),
        (),
    ):
        node = model
        ok = True
        for a in attr_chain:
            if not hasattr(node, a):
                ok = False
                break
            node = getattr(node, a)
        if not ok:
            continue
        for c in candidates:
            if hasattr(node, c):
                return getattr(node, c)
    return None  # type: ignore[return-value]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=str)
    ap.add_argument("--checkpoint", type=str, default=None)
    ap.add_argument("--override", nargs="*", default=[])
    ap.add_argument("--n_samples", type=int, default=4)
    ap.add_argument("--out", type=str, default="report/figures/saliency_panel.pdf")
    ap.add_argument(
        "--methods",
        nargs="+",
        default=["grad_x_input", "grad_cam"],
        choices=["grad_x_input", "grad_cam"],
    )
    args = ap.parse_args()

    cfg = load_config(args.config, overrides=args.override)
    set_seed(int(cfg.seed))

    device = torch.device(
        cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu"
    )

    model = build_model(cfg).to(device)
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        print(f"Loaded checkpoint from {args.checkpoint}")
    model.eval()

    loader = _build_eval_loader(cfg, args.n_samples)
    batch = next(iter(loader))
    x = batch["image"].to(device).float()
    y = batch["label"].to(device).long()

    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=-1)[:, 1]

    saliencies = []
    method_names = []
    if "grad_x_input" in args.methods:
        sal = gradient_times_input(model, x, target_class=1)
        saliencies.append(sal)
        method_names.append("grad x input")

    if "grad_cam" in args.methods:
        target = _resolve_target_module(model)
        if target is None:
            print(
                "[grad_cam] no suitable target module found for this model; "
                "skipping Grad-CAM (only gradient-times-input will be plotted)."
            )
        else:
            cam = GradCAM3D(model, target)
            try:
                sal = cam(x, target_class=1)
                saliencies.append(sal)
                method_names.append("3D Grad-CAM")
            finally:
                cam.close()

    save_saliency_panel(
        images=x,
        saliencies=saliencies,
        method_names=method_names,
        out_path=args.out,
        labels=y.cpu().tolist(),
        probs=probs.cpu().tolist(),
    )
    print(f"Saved saliency panel -> {args.out}")


if __name__ == "__main__":
    main()
