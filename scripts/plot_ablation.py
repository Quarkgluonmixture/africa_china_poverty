"""Overlay validation r² vs epoch for pretrained vs from-scratch ResNet-50.

Visualises the transfer-learning ablation: pretrained converges in a few epochs
to a higher plateau; from-scratch needs an order of magnitude more epochs.

    python scripts/plot_ablation.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/figures/01_pretrain_vs_scratch.png"


def main() -> None:
    pre = pd.read_csv(ROOT / "outputs/resnet50/history.csv")
    scr = pd.read_csv(ROOT / "outputs/resnet50_scratch/history.csv")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(pre["epoch"], pre["val_pearson_r2"], color="tab:blue", lw=2.5,
            label=f"ImageNet-pretrained (best {pre['val_pearson_r2'].max():.3f})")
    ax.plot(scr["epoch"], scr["val_pearson_r2"], color="tab:red", lw=2.5, ls="--",
            label=f"from scratch (best {scr['val_pearson_r2'].max():.3f})")
    ax.axhline(0.6, color="grey", lw=1, ls=":", label="_nolegend_")
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation r² (squared Pearson)")
    ax.set_title("Transfer-learning ablation — ResNet-50 on ~2k tiles\n"
                 "pretraining: higher plateau, ~10× faster convergence")
    ax.legend(loc="lower right")
    ax.set_ylim(-0.1, 0.8)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200)
    print(f"[OK] {OUT}")


if __name__ == "__main__":
    main()
