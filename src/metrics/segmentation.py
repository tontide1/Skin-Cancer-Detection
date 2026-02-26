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


@torch.no_grad()
def find_best_threshold(
    pred: torch.Tensor,
    target: torch.Tensor,
    thresholds: list[float] | None = None,
) -> tuple[float, float]:
    """
    Tìm threshold tối ưu maximize Dice trên validation set.

    Args:
        pred:        Raw logits (B, 1, H, W)
        target:      Binary mask (B, 1, H, W)
        thresholds:  List ngưỡng cần test (default: 0.3 → 0.7, step 0.05)

    Returns:
        (best_threshold, best_dice)
    """
    if thresholds is None:
        thresholds = [round(t, 2) for t in torch.arange(0.3, 0.71, 0.05).tolist()]

    best_thr, best_dice = 0.5, 0.0
    for thr in thresholds:
        d = dice_coefficient(pred, target, threshold=thr)
        if d > best_dice:
            best_dice = d
            best_thr  = thr

    return best_thr, best_dice
