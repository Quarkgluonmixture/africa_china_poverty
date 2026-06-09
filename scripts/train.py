"""Train a poverty-regression model on the DHS Africa imagery.

Example:

    python scripts/train.py --config configs/resnet50.yaml
    python scripts/train.py --config configs/convnext_tiny.yaml --epochs 40
    python scripts/train.py --config configs/resnet50.yaml --split country_holdout --holdout-country NG

Outputs (under outputs/<run_name>/): best.pt, history.csv, curve.png,
test_metrics.json, test_predictions.csv.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from acp.data import load_clusters, make_dataloaders, make_splits  # noqa: E402
from acp.engine import evaluate, fit  # noqa: E402
from acp.models import build_model  # noqa: E402
from acp.utils import amp_dtype, get_device, load_config, set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/resnet50.yaml")
    p.add_argument("--backbone")
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--img-size", type=int)
    p.add_argument("--split", choices=["random", "country_holdout"])
    p.add_argument("--holdout-country")
    p.add_argument("--workers", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--run-name")
    return p.parse_args()


def merge(cfg: dict, args: argparse.Namespace) -> dict:
    for k in ["backbone", "epochs", "batch_size", "lr", "img_size", "split",
              "holdout_country", "workers", "seed", "run_name"]:
        v = getattr(args, k, None)
        if v is not None:
            cfg[k] = v
    return cfg


def plot_curve(history: list[dict], out: Path, title: str) -> None:
    df = pd.DataFrame(history)
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(df["epoch"], df["train_loss"], color="tab:red", lw=2, label="train loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("train loss (MSE, scaled)", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red")
    ax2 = ax1.twinx()
    ax2.plot(df["epoch"], df["val_r2"], color="tab:blue", lw=2.5, label="val R²")
    ax2.plot(df["epoch"], df["val_pearson_r2"], color="tab:green", lw=2, ls="--", label="val r² (pearson)")
    ax2.set_ylabel("validation score", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")
    ax2.axhline(0.0, color="grey", lw=1, ls=":", label="_nolegend_")
    lines = [ln for ln in ax1.lines + ax2.lines if not ln.get_label().startswith("_")]
    ax1.legend(lines, [ln.get_label() for ln in lines], loc="lower right")
    plt.title(title)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    cfg = merge(load_config(args.config), args)
    cfg.setdefault("split", "random")
    cfg.setdefault("workers", 8)
    cfg.setdefault("seed", 42)
    cfg.setdefault("run_name", cfg["backbone"])
    cfg.setdefault("clusters_csv", "data/clusters.csv")
    cfg.setdefault("images_dir", "data/images")

    set_seed(cfg["seed"])
    device = get_device()
    dtype = amp_dtype(device)
    print(f"device={device} amp_dtype={dtype} config={args.config}")
    print(json.dumps(cfg, indent=2))

    out_dir = ROOT / "outputs" / cfg["run_name"]
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_clusters(cfg["clusters_csv"], cfg["images_dir"])
    splits = make_splits(
        df, scheme=cfg["split"], seed=cfg["seed"],
        holdout_country=cfg.get("holdout_country"),
    )
    print(f"data: total={len(df)} | train={len(splits[0])} val={len(splits[1])} test={len(splits[2])}")
    loaders, scaler = make_dataloaders(
        df, splits, img_size=cfg["img_size"],
        batch_size=cfg["batch_size"], num_workers=cfg["workers"],
    )

    model = build_model(cfg["backbone"], pretrained=cfg.get("pretrained", True),
                        drop_rate=cfg.get("drop_rate", 0.2)).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"model={cfg['backbone']} pretrained={cfg.get('pretrained', True)} params={n_params:.1f}M")

    history = fit(
        model, loaders, scaler, device, dtype,
        epochs=cfg["epochs"], lr=cfg["lr"],
        weight_decay=cfg.get("weight_decay", 0.05),
        warmup_epochs=cfg.get("warmup_epochs", 3),
        patience=cfg.get("patience", 12),
        ckpt_path=out_dir / "best.pt",
        log_csv=out_dir / "history.csv",
    )

    # Restore best and evaluate on the held-out test set.
    import torch

    state = torch.load(out_dir / "best.pt", map_location=device)
    model.load_state_dict(state["model"])
    test_metrics, ids, y_true, y_pred = evaluate(model, loaders["test"], device, dtype, scaler)
    print("TEST:", json.dumps({k: round(v, 4) for k, v in test_metrics.items()}))

    (out_dir / "test_metrics.json").write_text(json.dumps(
        {"config": cfg, "best_epoch": state["epoch"],
         "val_metrics": state["val_metrics"], "test_metrics": test_metrics}, indent=2))
    pd.DataFrame({"unique_id": ids, "actual": y_true, "predicted": y_pred}).to_csv(
        out_dir / "test_predictions.csv", index=False)
    plot_curve(history, out_dir / "curve.png",
               f"{cfg['backbone']} (ImageNet-pretrained) — real training curve\n"
               f"test R²={test_metrics['r2']:.3f}  r²(pearson)={test_metrics['pearson_r2']:.3f}")
    print(f"[OK] all artifacts in {out_dir}")


if __name__ == "__main__":
    main()
