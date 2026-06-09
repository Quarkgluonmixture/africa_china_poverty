"""Fast end-to-end pipeline check on a tiny subset (no full training).

Validates data loading, model build (ImageNet download), one AMP train step,
and evaluation — then reports peak GPU memory so we know how much head-room the
shared GB10 has right now.

    python scripts/smoke_test.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from acp.data import load_clusters, make_dataloaders, make_splits  # noqa: E402
from acp.engine import evaluate, train_one_epoch  # noqa: E402
from acp.models import build_model  # noqa: E402
from acp.utils import amp_dtype, get_device, set_seed  # noqa: E402


def main() -> None:
    set_seed(0)
    device = get_device()
    dtype = amp_dtype(device)
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info()
        print(f"GPU free={free/1e9:.1f}GB / total={total/1e9:.1f}GB  amp={dtype}")

    df = load_clusters("data/clusters.csv", "data/images")
    # tiny subset for speed
    df = df.groupby("country", group_keys=False).head(40).reset_index(drop=True)
    splits = make_splits(df, scheme="random", seed=0)
    loaders, scaler = make_dataloaders(df, splits, img_size=224, batch_size=16, num_workers=4)
    print(f"subset total={len(df)} train={len(splits[0])} val={len(splits[1])} test={len(splits[2])}")

    model = build_model("resnet50", pretrained=True).to(device)
    print("model built (resnet50, ImageNet weights downloaded)")

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    loss_fn = torch.nn.MSELoss()
    t0 = time.time()
    loss = train_one_epoch(model, loaders["train"], opt, None, device, dtype, loss_fn)
    metrics, *_ = evaluate(model, loaders["val"], device, dtype, scaler)
    dt = time.time() - t0

    print(f"1 train epoch loss={loss:.4f} | val R²={metrics['r2']:+.4f} | {dt:.1f}s")
    if device.type == "cuda":
        print(f"peak GPU mem this run = {torch.cuda.max_memory_allocated()/1e9:.2f}GB")
    print("[OK] pipeline works end to end")


if __name__ == "__main__":
    main()
