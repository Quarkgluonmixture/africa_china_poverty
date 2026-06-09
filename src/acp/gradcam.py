"""Minimal, self-contained Grad-CAM for the regression model.

The CAM is the gradient-weighted activation map of a trained model w.r.t. its
scalar wealth prediction. For a single-output regression head the "score" we
differentiate is just the predicted value, so the CAM highlights the regions
that push the predicted wealth index up.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def last_conv_layer(model: nn.Module) -> nn.Module:
    """Return the last Conv2d module — a sensible default CAM target for CNNs."""
    conv = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            conv = m
    if conv is None:
        raise ValueError("no Conv2d found; pass an explicit target layer (e.g. for ViT)")
    return conv


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module | None = None) -> None:
        self.model = model.eval()
        self.target = target_layer or last_conv_layer(model)
        self._act: torch.Tensor | None = None
        self._grad: torch.Tensor | None = None
        self.target.register_forward_hook(self._save_act)
        self.target.register_full_backward_hook(self._save_grad)

    def _save_act(self, _m, _inp, out):
        self._act = out.detach()

    def _save_grad(self, _m, _gin, gout):
        self._grad = gout[0].detach()

    def __call__(self, x: torch.Tensor) -> tuple[np.ndarray, float]:
        """x: (1, C, H, W). Returns (cam[H,W] in [0,1], predicted_value)."""
        self.model.zero_grad(set_to_none=True)
        out = self.model(x)                 # (1, 1)
        score = out.squeeze()
        score.backward()
        # weights: global-average-pooled gradients over spatial dims
        weights = self._grad.mean(dim=(2, 3), keepdim=True)        # (1, K, 1, 1)
        cam = F.relu((weights * self._act).sum(dim=1, keepdim=True))  # (1, 1, h, w)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, float(score.detach().cpu())
