"""Zero-shot transfer: apply an Africa-trained model to the 20 Guizhou tiles.

Images live in ``china_dataset_final/`` named ``{label}_{Location}.jpg`` where
label 1 = developed/urban, 0 = poor/rural. Predictions are inverse-transformed
to wealth-index units using the scaler saved in the checkpoint, so they are
directly comparable to the African labels.

    python scripts/predict_china.py --ckpt outputs/resnet50/best.pt

Outputs: outputs/<run>/china_predictions.csv and china_boxplot.png.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from acp.data import build_transforms  # noqa: E402
from acp.models import build_model  # noqa: E402
from acp.utils import TargetScaler, amp_dtype, get_device  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="path to best.pt from training")
    p.add_argument("--backbone", required=True, help="must match the checkpoint")
    p.add_argument("--china-dir", default="china_dataset_final")
    p.add_argument("--img-size", type=int, default=224)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()
    dtype = amp_dtype(device)

    state = torch.load(args.ckpt, map_location=device)
    scaler = TargetScaler(**state["scaler"])
    model = build_model(args.backbone, pretrained=False).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    tf = build_transforms(args.img_size, train=False)
    china_dir = Path(args.china_dir)
    files = sorted(f for f in china_dir.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not files:
        raise SystemExit(f"no images in {china_dir}")

    rows = []
    for f in files:
        label = int(f.name[0])
        location = f.stem[2:] if f.stem[1] == "_" else f.stem[1:]
        img = tf(Image.open(f).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad(), torch.autocast(device.type, dtype=dtype, enabled=device.type == "cuda"):
            pred_scaled = model(img).float().cpu().item()
        rows.append({"image": f.name, "location": location, "true_label": label,
                     "predicted_wealth": scaler.inverse(np.array([pred_scaled]))[0]})

    df = pd.DataFrame(rows).sort_values("predicted_wealth", ascending=False)
    out_dir = Path(args.ckpt).parent
    df.to_csv(out_dir / "china_predictions.csv", index=False)

    rich = df[df.true_label == 1]["predicted_wealth"]
    poor = df[df.true_label == 0]["predicted_wealth"]
    gap = rich.mean() - poor.mean()
    print(df[["location", "true_label", "predicted_wealth"]].to_string(index=False))
    print(f"\nmean(developed)={rich.mean():.3f}  mean(poor)={poor.mean():.3f}  gap={gap:.3f}")

    fig, ax = plt.subplots(figsize=(7, 6))
    data = [poor.values, rich.values]
    ax.boxplot(data, tick_labels=["poor / rural (0)", "developed / urban (1)"], showfliers=False)
    for i, vals in enumerate(data, start=1):
        ax.scatter(np.full(len(vals), i) + np.random.uniform(-0.05, 0.05, len(vals)),
                   vals, color="0.25", zorder=3)
    ax.set_ylabel("predicted wealth index")
    ax.set_title(f"Zero-shot transfer to Guizhou ({args.backbone})\ngap = {gap:.2f}")
    fig.tight_layout()
    fig.savefig(out_dir / "china_boxplot.png", dpi=200)
    print(f"[OK] {out_dir}/china_predictions.csv, china_boxplot.png")


if __name__ == "__main__":
    main()
