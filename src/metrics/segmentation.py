"""
Evaluation metrics cho binary segmentation.
"""

from __future__ import annotations

import torch


@torch.no_grad()
def dice_coefficient(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    """
    Dice Coefficient (macro-average: tính per-image rồi trung bình).

    Args:
        pred:      Raw logits (B, 1, H, W)
        target:    Binary mask (B, 1, H, W)
        threshold: Ngưỡng binarize predictions

    Returns:
        float trong [0, 1]
    """
    probs  = torch.sigmoid(pred)
    binary = (probs > threshold).float()

    flat_pred   = binary.view(binary.size(0), -1)
    flat_target = target.view(target.size(0), -1).float()

    intersection = (flat_pred * flat_target).sum(1)
    union        = flat_pred.sum(1) + flat_target.sum(1)
    dice         = (2.0 * intersection + 1e-7) / (union + 1e-7)

    return dice.mean().item()


@torch.no_grad()
def iou_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    """
    IoU / Jaccard Score (macro-average).

    Args:
        pred:      Raw logits (B, 1, H, W)
        target:    Binary mask (B, 1, H, W)
        threshold: Ngưỡng binarize predictions

    Returns:
        float trong [0, 1]
    """
    probs  = torch.sigmoid(pred)
    binary = (probs > threshold).float()

    flat_pred   = binary.view(binary.size(0), -1)
    flat_target = target.view(target.size(0), -1).float()

    intersection = (flat_pred * flat_target).sum(1)
    union        = flat_pred.sum(1) + flat_target.sum(1) - intersection
    iou          = (intersection + 1e-7) / (union + 1e-7)

    return iou.mean().item()

