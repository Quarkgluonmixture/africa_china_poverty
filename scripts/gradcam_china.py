"""Generate real Grad-CAM overlays for the Guizhou tiles using a trained model.

    python scripts/gradcam_china.py --ckpt outputs/resnet50/best.pt --backbone resnet50

Saves one overlay per image plus a contact-sheet to outputs/<run>/gradcam/.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from acp.data import build_transforms  # noqa: E402
from acp.gradcam import GradCAM  # noqa: E402
from acp.models import build_model  # noqa: E402
from acp.utils import TargetScaler, get_device  # noqa: E402


def overlay(img: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    heat = cm.jet(cam)[..., :3]
    return np.clip((1 - alpha) * img + alpha * heat, 0, 1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--backbone", required=True)
    p.add_argument("--china-dir", default="china_dataset_final")
    p.add_argument("--img-size", type=int, default=224)
    args = p.parse_args()

    device = get_device()
    state = torch.load(args.ckpt, map_location=device)
    scaler = TargetScaler(**state["scaler"])
    model = build_model(args.backbone, pretrained=False).to(device)
    model.load_state_dict(state["model"])
    cam_fn = GradCAM(model)
    tf = build_transforms(args.img_size, train=False)

    out_dir = Path(args.ckpt).parent / "gradcam"
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(f for f in Path(args.china_dir).iterdir()
                   if f.suffix.lower() in {".jpg", ".jpeg", ".png"})

    n = len(files)
    cols = 5
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.2))
    axes = np.atleast_1d(axes).ravel()

    for ax, f in zip(axes, files):
        pil = Image.open(f).convert("RGB").resize((args.img_size, args.img_size))
        base = np.asarray(pil, dtype=np.float32) / 255.0
        x = tf(Image.open(f).convert("RGB")).unsqueeze(0).to(device)
        cam, score = cam_fn(x)
        ov = overlay(base, cam)
        label = int(f.name[0])
        wealth = scaler.inverse(np.array([score]))[0]
        ax.imshow(ov)
        ax.set_title(f"{f.stem[2:][:18]}\nlabel={label} pred={wealth:.2f}", fontsize=8)
        ax.axis("off")
        Image.fromarray((ov * 255).astype(np.uint8)).save(out_dir / f"{f.stem}_cam.png")
    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(f"Grad-CAM on Guizhou tiles — trained {args.backbone}", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "contact_sheet.png", dpi=150)
    print(f"[OK] {out_dir}/ (per-image overlays + contact_sheet.png)")


if __name__ == "__main__":
    main()
