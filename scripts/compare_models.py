"""Aggregate test metrics across backbones into a comparison table + bar chart.

Reads every ``outputs/*/test_metrics.json`` produced by ``train.py`` and writes
``outputs/model_comparison.csv`` and ``outputs/model_comparison.png``.

    python scripts/compare_models.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"


def main() -> None:
    rows = []
    for mj in sorted(OUT.glob("*/test_metrics.json")):
        d = json.loads(mj.read_text())
        tm = d["test_metrics"]
        rows.append({
            "run": mj.parent.name,
            "backbone": d["config"].get("backbone"),
            "split": d["config"].get("split"),
            "best_epoch": d.get("best_epoch"),
            "test_r2": round(tm["r2"], 4),
            "test_pearson_r2": round(tm["pearson_r2"], 4),
            "test_rmse": round(tm["rmse"], 4),
        })
    if not rows:
        raise SystemExit("no outputs/*/test_metrics.json yet — train a model first")

    df = pd.DataFrame(rows).sort_values("test_pearson_r2", ascending=False)
    df.to_csv(OUT / "model_comparison.csv", index=False)
    print(df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(8, 5))
    # Colour the from-scratch ablation differently from the pretrained models.
    colors = ["tab:red" if "scratch" in r else "tab:blue" for r in df["run"]]
    ax.barh(df["run"], df["test_pearson_r2"], color=colors)
    ax.set_xlabel("test r² (squared Pearson)")
    ax.set_title("Backbone comparison — Africa DHS wealth regression")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color="tab:blue", label="ImageNet-pretrained"),
        plt.Rectangle((0, 0), 1, 1, color="tab:red", label="from scratch (ablation)"),
    ]
    ax.legend(handles=handles, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "model_comparison.png", dpi=200)
    print(f"[OK] {OUT}/model_comparison.csv, model_comparison.png")


if __name__ == "__main__":
    main()
