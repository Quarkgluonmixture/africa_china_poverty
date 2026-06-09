"""Re-plot a training curve from a run's history.csv (no retraining needed).

    python scripts/plot_history.py outputs/resnet50 [--out path.png]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_curve(history: list[dict], out: Path, title: str) -> None:
    df = pd.DataFrame(history)
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(df["epoch"], df["train_loss"], color="tab:red", lw=2, label="train loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("train loss (MSE, scaled)", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red")
    ax2 = ax1.twinx()
    ax2.plot(df["epoch"], df["val_r2"], color="tab:blue", lw=2.5, label="val R²")
    ax2.plot(df["epoch"], df["val_pearson_r2"], color="tab:green", lw=2, ls="--",
             label="val r² (pearson)")
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
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", help="e.g. outputs/resnet50")
    p.add_argument("--out")
    args = p.parse_args()
    run = Path(args.run_dir)
    hist = pd.read_csv(run / "history.csv").to_dict("records")
    meta = json.loads((run / "test_metrics.json").read_text())
    tm, cfg = meta["test_metrics"], meta["config"]
    out = Path(args.out) if args.out else run / "curve.png"
    plot_curve(hist, out,
               f"{cfg['backbone']} (ImageNet-pretrained) — real training curve\n"
               f"test R²={tm['r2']:.3f}  r²(pearson)={tm['pearson_r2']:.3f}")
    print(f"[OK] {out}")


if __name__ == "__main__":
    main()
