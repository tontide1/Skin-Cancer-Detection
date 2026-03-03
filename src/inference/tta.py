"""Test-Time Augmentation utilities for segmentation inference."""

from __future__ import annotations

import torch
import torch.nn as nn


@torch.no_grad()
def tta_predict(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    """
    Test-Time Augmentation: average sigmoid probs over original + hflip + vflip.

    Args:
        model:  Segmentation model returning raw logits (B, C, H, W).
        images: Input tensor (B, C, H, W).

    Returns:
        Averaged probabilities in [0, 1], shape (B, C, H, W).
    """
    probs = torch.sigmoid(model(images))
    probs += torch.sigmoid(model(torch.flip(images, dims=[3]))).flip(dims=[3])
    probs += torch.sigmoid(model(torch.flip(images, dims=[2]))).flip(dims=[2])
    return probs / 3.0
