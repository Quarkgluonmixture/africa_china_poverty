"""Small shared utilities: reproducibility, device selection, config loading."""
from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def set_seed(seed: int = 42) -> None:
    """Seed python, numpy and torch for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def amp_dtype(device: torch.device) -> torch.dtype:
    """bf16 on modern CUDA (Blackwell GB10 supports it natively), else fp16/fp32."""
    if device.type != "cuda":
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


@dataclass
class TargetScaler:
    """Standardise the regression target with train-split statistics.

    R²/Pearson are scale-invariant, but standardising the target stabilises the
    MSE loss. Predictions are always inverse-transformed before metrics so every
    number is reported in the original wealth-index units.
    """

    mean: float = 0.0
    std: float = 1.0

    def transform(self, y: np.ndarray) -> np.ndarray:
        return (y - self.mean) / self.std

    def inverse(self, y: np.ndarray) -> np.ndarray:
        return y * self.std + self.mean
