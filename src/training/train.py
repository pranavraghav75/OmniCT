"""Training entry point.

Run:
    python -m src.training.train \
        --config src/configs/baseline_cnn3d.yaml \
        --override seed=0 train.epochs=10

This script:
    1. Loads the YAML config and applies CLI overrides.
    2. Seeds all RNGs.
    3. Builds dataset, model, optimizer, scheduler.
    4. Runs train/val loops, logging to stdout + a per-run log file.
    5. On finish, runs the held-out evaluation and writes
       `results/<run_name>/metrics.json` plus the resolved config.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data import (
    OmniCTDataset,
    SyntheticCTDataset,
    build_eval_transforms,
    build_train_transforms,
)
from src.models import build_model
from src.training.metrics import compute_classification_metrics
from src.utils import get_logger, load_config, save_config, set_seed


def _build_dataloaders(cfg) -> tuple[DataLoader, DataLoader, DataLoader | None]:
    if cfg.data.synthetic:
        train_ds = SyntheticCTDataset(
            n_samples=int(cfg.data.synthetic_n_train),
            spatial_size=tuple(cfg.data.spatial_size),
            n_classes=int(cfg.model.n_classes),
            seed=int(cfg.seed),
        )
        val_ds = SyntheticCTDataset(
            n_samples=int(cfg.data.synthetic_n_val),
            spatial_size=tuple(cfg.data.spatial_size),
            n_classes=int(cfg.model.n_classes),
            seed=int(cfg.seed) + 1,
        )
        test_ds = None
    else:
        train_tf = build_train_transforms(
            spacing=tuple(cfg.data.spacing),
            hu_window=tuple(cfg.data.hu_window),
            spatial_size=tuple(cfg.data.spatial_size),
        )
        eval_tf = build_eval_transforms(
            spacing=tuple(cfg.data.spacing),
            hu_window=tuple(cfg.data.hu_window),
            spatial_size=tuple(cfg.data.spatial_size),
        )

        def _ids(path):
            p = Path(path)
            return p.read_text().strip().splitlines() if p.exists() else None

        train_ds = OmniCTDataset(
            manifest=cfg.data.manifest,
            data_root=cfg.data.data_root,
            split_ids=_ids(cfg.data.train_ids),
            transform=train_tf,
        )
        val_ds = OmniCTDataset(
            manifest=cfg.data.manifest,
            data_root=cfg.data.data_root,
            split_ids=_ids(cfg.data.val_ids),
            transform=eval_tf,
        )
        test_ids = _ids(cfg.data.test_ids)
        test_ds = (
            OmniCTDataset(
                manifest=cfg.data.manifest,
                data_root=cfg.data.data_root,
                split_ids=test_ids,
                transform=eval_tf,
            )
            if test_ids
            else None
        )

    bs = int(cfg.train.batch_size)
    nw = int(cfg.train.num_workers)
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw, drop_last=True),
        DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw),
        DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=nw) if test_ds is not None else None,
    )


def _evaluate(model, loader, device) -> dict[str, float]:
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device, non_blocking=True).float()
            y = batch["label"].to(device, non_blocking=True).long()
            logits = model(x)
            prob_pos = F.softmax(logits, dim=-1)[:, 1]
            ys.extend(y.cpu().tolist())
            ps.extend(prob_pos.cpu().tolist())
    return compute_classification_metrics(ys, ps)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--override", nargs="*", default=[])
    parser.add_argument("--run_name", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config, overrides=args.override)
    run_name = args.run_name or f"{cfg.run_name}_seed{int(cfg.seed)}"
    out_dir = Path(cfg.output_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger("omnict.train", log_file=out_dir / "train.log")
    logger.info("Run: %s", run_name)
    logger.info("Output dir: %s", out_dir)

    set_seed(int(cfg.seed), deterministic=bool(cfg.deterministic))
    save_config(cfg, out_dir / "config.yaml")

    device = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu")
    logger.info("Using device: %s", device)

    train_loader, val_loader, test_loader = _build_dataloaders(cfg)
    logger.info("Datasets — train=%d, val=%d, test=%s",
                len(train_loader.dataset),
                len(val_loader.dataset),
                len(test_loader.dataset) if test_loader else "None")

    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model: %s — trainable params=%d", cfg.model.kind, n_params)

    # Class-prior baseline doesn't train.
    if cfg.model.kind == "prior":
        labels = torch.tensor([s["label"] for s in train_loader.dataset])
        model.fit_prior(labels)
        val_metrics = _evaluate(model, val_loader, device)
        test_metrics = _evaluate(model, test_loader, device) if test_loader else {}
        (out_dir / "metrics.json").write_text(
            json.dumps({"val": val_metrics, "test": test_metrics}, indent=2)
        )
        logger.info("Prior baseline — val: %s", val_metrics)
        return

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(cfg.train.lr),
        weight_decay=float(cfg.train.weight_decay),
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=int(cfg.train.epochs))

    best_val_auc = -np.inf
    best_path = out_dir / "best.pt"

    for epoch in range(int(cfg.train.epochs)):
        model.train()
        t0 = time.time()
        running = 0.0
        n = 0
        for batch in train_loader:
            x = batch["image"].to(device, non_blocking=True).float()
            y = batch["label"].to(device, non_blocking=True).long()

            logits = model(x)
            loss = F.cross_entropy(logits, y)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=float(cfg.train.grad_clip),
            )
            optim.step()

            running += float(loss.item()) * x.shape[0]
            n += x.shape[0]
        sched.step()

        train_loss = running / max(n, 1)
        val_metrics = _evaluate(model, val_loader, device)
        logger.info(
            "epoch %3d | train_loss=%.4f | val_auc=%.4f | val_f1=%.4f | val_bal=%.4f | %.1fs",
            epoch, train_loss,
            val_metrics["roc_auc"], val_metrics["f1"], val_metrics["balanced_accuracy"],
            time.time() - t0,
        )

        if val_metrics["roc_auc"] > best_val_auc:
            best_val_auc = val_metrics["roc_auc"]
            torch.save({"state_dict": model.state_dict(), "epoch": epoch, "val": val_metrics}, best_path)

    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device)["state_dict"])

    final = {"val": _evaluate(model, val_loader, device)}
    if test_loader is not None:
        final["test"] = _evaluate(model, test_loader, device)
    (out_dir / "metrics.json").write_text(json.dumps(final, indent=2))
    logger.info("Done. Final metrics: %s", final)


if __name__ == "__main__":
    main()
