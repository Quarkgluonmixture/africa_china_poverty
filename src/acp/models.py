"""Model factory: ImageNet-pretrained backbones with a scalar regression head.

Using ``timm`` gives a single API over the modern backbones we want to compare
(ResNet-50 baseline, ConvNeXt, ViT). ``num_classes=1`` swaps the classifier for
a single linear output suitable for wealth-index regression.

The key fix over the original from-scratch run: ``pretrained=True``. Fine-tuning
ImageNet features is what makes a ~2k-image regression task learnable at all.
"""
from __future__ import annotations

import timm
import torch.nn as nn


def build_model(
    name: str = "resnet50",
    pretrained: bool = True,
    drop_rate: float = 0.2,
) -> nn.Module:
    """Create a backbone with a 1-unit regression head.

    Args:
        name: any ``timm`` model, e.g. ``resnet50``, ``convnext_tiny``,
            ``vit_small_patch16_224``.
        pretrained: load ImageNet weights (strongly recommended).
        drop_rate: dropout before the head, for regularisation on small data.
    """
    model = timm.create_model(
        name,
        pretrained=pretrained,
        num_classes=1,
        drop_rate=drop_rate,
    )
    return model
