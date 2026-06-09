"""Training / evaluation loops, metrics, early stopping.

All reported metrics are computed in the original wealth-index units (the
``TargetScaler`` is inverted on predictions first):
  - ``r2``        : coefficient of determination (1 - SS_res/SS_tot)
  - ``pearson_r2``: squared Pearson correlation (the metric Yeh et al. report)
  - ``rmse`` / ``mae``
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .utils import TargetScaler


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / (ss_tot + 1e-12)
    if y_pred.std() < 1e-9 or y_true.std() < 1e-9:
        pearson = 0.0
    else:
        pearson = float(np.corrcoef(y_true, y_pred)[0, 1])
    return {
        "r2": r2,
        "pearson_r2": pearson**2,
        "rmse": math.sqrt(ss_res / len(y_true)),
        "mae": float(np.mean(np.abs(y_true - y_pred))),
    }


def train_one_epoch(model, loader, optimizer, scheduler, device, amp_dtype, loss_fn):
    model.train()
    use_scaler = amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    running = 0.0
    for imgs, targets, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True).unsqueeze(1)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
            preds = model(imgs)
            loss = loss_fn(preds, targets)
        if use_scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
        running += loss.item() * imgs.size(0)
    return running / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device, amp_dtype, scaler: TargetScaler):
    """Return (metrics, ids, y_true, y_pred) in original units."""
    model.eval()
    ids, preds_all, tgts_all = [], [], []
    for imgs, targets, batch_ids in loader:
        imgs = imgs.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
            preds = model(imgs).float().cpu().numpy().ravel()
        preds_all.append(preds)
        tgts_all.append(targets.numpy().ravel())
        ids.extend(batch_ids)
    y_pred_scaled = np.concatenate(preds_all)
    y_true_scaled = np.concatenate(tgts_all)
    y_pred = scaler.inverse(y_pred_scaled)
    y_true = scaler.inverse(y_true_scaled)
    return compute_metrics(y_true, y_pred), ids, y_true, y_pred


def fit(
    model,
    loaders: dict[str, DataLoader],
    scaler: TargetScaler,
    device,
    amp_dtype,
    epochs: int = 50,
    lr: float = 3e-4,
    weight_decay: float = 0.05,
    warmup_epochs: int = 3,
    patience: int = 12,
    ckpt_path: str | Path = "checkpoints/best.pt",
    log_csv: str | Path | None = None,
    select_metric: str = "pearson_r2",
):
    """Train with AdamW + cosine schedule (linear warmup) and early stopping on
    the validation ``select_metric``. Returns the per-epoch history (list of dicts).
    """
    ckpt_path = Path(ckpt_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    steps_per_epoch = max(1, len(loaders["train"]))
    total_steps = epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * prog))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    history: list[dict] = []
    best_score = -float("inf")
    bad_epochs = 0

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model, loaders["train"], optimizer, scheduler, device, amp_dtype, loss_fn
        )
        val_metrics, *_ = evaluate(model, loaders["val"], device, amp_dtype, scaler)
        row = {"epoch": epoch, "train_loss": train_loss,
               "lr": optimizer.param_groups[0]["lr"], **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        score = val_metrics[select_metric]
        flag = ""
        if score > best_score:
            best_score = score
            bad_epochs = 0
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "val_metrics": val_metrics,
                        "scaler": {"mean": scaler.mean, "std": scaler.std}}, ckpt_path)
            flag = "  * best"
        else:
            bad_epochs += 1
        print(
            f"epoch {epoch:3d}/{epochs} | train_loss {train_loss:.4f} | "
            f"val R² {val_metrics['r2']:+.4f} | val r²(pearson) {val_metrics['pearson_r2']:.4f} | "
            f"RMSE {val_metrics['rmse']:.4f}{flag}"
        )
        if log_csv is not None:
            _append_csv(log_csv, row)
        if bad_epochs >= patience:
            print(f"early stopping at epoch {epoch} (no val {select_metric} gain in {patience})")
            break

    print(f"best val {select_metric} = {best_score:.4f} -> {ckpt_path}")
    return history


def _append_csv(path: str | Path, row: dict) -> None:
    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)
