"""Test-Time Augmentation utilities for segmentation inference."""

from __future__ import annotations

import torch
import torch.nn as nn


@torch.no_grad()
def tta_predict(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    """
    Test-Time Augmentation: average sigmoid probs over 5 geometric views.

    Views:
        1. Original
        2. Horizontal flip
        3. Vertical flip
        4. 90° rotation  (rot90 k=1)
        5. 270° rotation (rot90 k=3)

    Dermoscopy images are rotationally symmetric (no canonical orientation),
    so all 5 views are valid and improve boundary delineation.

    Args:
        model:  Segmentation model returning raw logits (B, C, H, W).
        images: Input tensor (B, C, H, W).

    Returns:
        Averaged probabilities in [0, 1], shape (B, C, H, W).
    """
    # Original
    probs = torch.sigmoid(model(images))

    # Horizontal flip (along W)
    probs += torch.sigmoid(model(torch.flip(images, dims=[3]))).flip(dims=[3])

    # Vertical flip (along H)
    probs += torch.sigmoid(model(torch.flip(images, dims=[2]))).flip(dims=[2])

    # 90° rotation — rotate input, un-rotate prediction
    probs += torch.sigmoid(model(torch.rot90(images, k=1, dims=[2, 3]))).rot90(k=3, dims=[2, 3])

    # 270° rotation — rotate input, un-rotate prediction
    probs += torch.sigmoid(model(torch.rot90(images, k=3, dims=[2, 3]))).rot90(k=1, dims=[2, 3])

    return probs / 5.0
