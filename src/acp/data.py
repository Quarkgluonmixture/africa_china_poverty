"""Data pipeline: DHS cluster wealth-index regression from RGB satellite tiles.

Labels come from ``data/clusters.csv`` (columns: unique_id, cluster_id,
wealth_index, LATNUM, LONGNUM, country); each cluster maps to
``data/images/{unique_id}.jpg``. Only ~2006 of the 3136 clusters have a
downloaded image, so we filter to what is present.

The original paper's ``dhs_incountry_folds.pkl`` indexes the *full* DHS dataset
(~15.7k clusters) and does not map onto this subset, so we build our own
reproducible splits here: a country-stratified random split ("in-country"
evaluation) or a leave-one-country-out split ("out-of-country").
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

from .utils import TargetScaler

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_clusters(clusters_csv: str | Path, images_dir: str | Path) -> pd.DataFrame:
    """Load cluster labels and keep only rows whose image file exists."""
    images_dir = Path(images_dir)
    df = pd.read_csv(clusters_csv)
    df["img_path"] = df["unique_id"].map(lambda u: images_dir / f"{u}.jpg")
    present = df["img_path"].map(lambda p: p.exists())
    kept = df[present].reset_index(drop=True)
    if len(kept) == 0:
        raise RuntimeError(f"No images found under {images_dir} for {clusters_csv}")
    return kept


def make_splits(
    df: pd.DataFrame,
    scheme: str = "random",
    seed: int = 42,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    holdout_country: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (train, val, test) DataFrames.

    scheme="random": country-stratified random split (in-country evaluation).
    scheme="country_holdout": ``holdout_country`` is the test set; the rest is
        split into train/val (out-of-country evaluation).
    """
    if scheme == "country_holdout":
        if holdout_country is None:
            raise ValueError("country_holdout requires holdout_country")
        test = df[df["country"] == holdout_country].reset_index(drop=True)
        rest = df[df["country"] != holdout_country].reset_index(drop=True)
        train, val = train_test_split(
            rest, test_size=val_frac, random_state=seed, stratify=rest["country"]
        )
        return train.reset_index(drop=True), val.reset_index(drop=True), test

    if scheme == "random":
        train, temp = train_test_split(
            df, test_size=val_frac + test_frac, random_state=seed, stratify=df["country"]
        )
        rel_test = test_frac / (val_frac + test_frac)
        val, test = train_test_split(
            temp, test_size=rel_test, random_state=seed, stratify=temp["country"]
        )
        return (
            train.reset_index(drop=True),
            val.reset_index(drop=True),
            test.reset_index(drop=True),
        )

    raise ValueError(f"unknown split scheme: {scheme}")


def build_transforms(img_size: int = 224, train: bool = True) -> v2.Compose:
    """ImageNet-style transforms; strong-but-sane augmentation for training."""
    if train:
        return v2.Compose(
            [
                v2.RandomResizedCrop(img_size, scale=(0.7, 1.0), antialias=True),
                v2.RandomHorizontalFlip(),
                v2.RandomVerticalFlip(),
                v2.RandomApply([v2.ColorJitter(0.2, 0.2, 0.2, 0.05)], p=0.5),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return v2.Compose(
        [
            v2.Resize(int(img_size * 1.14), antialias=True),
            v2.CenterCrop(img_size),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class PovertyDataset(Dataset):
    """Returns (image_tensor, scaled_target, unique_id)."""

    def __init__(
        self,
        df: pd.DataFrame,
        transform: v2.Compose,
        scaler: TargetScaler,
        target_col: str = "wealth_index",
    ) -> None:
        self.paths = df["img_path"].tolist()
        self.ids = df["unique_id"].tolist()
        self.targets = df[target_col].to_numpy(dtype=np.float32)
        self.scaled = scaler.transform(self.targets).astype(np.float32)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        img = Image.open(self.paths[i]).convert("RGB")
        img = self.transform(img)
        return img, torch.tensor(self.scaled[i]), self.ids[i]


def make_dataloaders(
    df: pd.DataFrame,
    splits: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    img_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 8,
) -> tuple[dict[str, DataLoader], TargetScaler]:
    train_df, val_df, test_df = splits
    scaler = TargetScaler(
        mean=float(train_df["wealth_index"].mean()),
        std=float(train_df["wealth_index"].std()),
    )
    loaders = {}
    for name, sdf, is_train in [
        ("train", train_df, True),
        ("val", val_df, False),
        ("test", test_df, False),
    ]:
        ds = PovertyDataset(sdf, build_transforms(img_size, is_train), scaler)
        loaders[name] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=is_train,
            persistent_workers=num_workers > 0,
        )
    return loaders, scaler
