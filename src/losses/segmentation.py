"""
Loss functions cho binary segmentation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss: FL(p_t) = (1 - p_t)^gamma * BCE(p_t)
    Tập trung vào hard examples (pixel biên/rìa).

    Args:
        gamma: Focusing parameter (default=2.0)
        alpha: Class weight cho lesion class. None = không weight.
    """

    def __init__(self, gamma: float = 2.0, alpha: float | None = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # inputs: raw logits (B, 1, H, W)
        # targets: binary mask {0, 1} (B, 1, H, W)
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-bce)
        focal = (1 - pt) ** self.gamma * bce

        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal = alpha_t * focal

        return focal.mean()


class SoftDiceLoss(nn.Module):
    """
    Soft Dice Loss — xử lý class imbalance tốt.
    Tính per-image rồi average → ảnh nhỏ không bị lấn át ảnh lớn.

    Args:
        eps: Smoothing factor tránh chia 0
    """

    def __init__(self, eps: float = 1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs   = torch.sigmoid(inputs)
        targets = targets.float()
        dims    = (1, 2, 3)  # tính trên H, W, C cho mỗi ảnh

        intersection = (probs * targets).sum(dims)
        union        = probs.sum(dims) + targets.sum(dims)
        dice_score   = (2.0 * intersection + self.eps) / (union + self.eps)

        return 1.0 - dice_score.mean()


class CombinedLoss(nn.Module):
    """
    Combined Loss = focal_weight * FocalLoss + dice_weight * SoftDiceLoss

    Config:
        config.training.loss.focal_weight  (default 0.5)
        config.training.loss.dice_weight   (default 0.5)
        config.training.loss.focal_gamma   (default 2.0)
        config.training.loss.focal_alpha   (default null)
    """

    def __init__(self, config):
        super().__init__()
        loss_cfg = config.training.loss
        self.focal_weight = loss_cfg.focal_weight
        self.dice_weight  = loss_cfg.dice_weight
        self.focal = FocalLoss(
            gamma=loss_cfg.focal_gamma,
            alpha=loss_cfg.focal_alpha,
        )
        self.dice = SoftDiceLoss()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return (
            self.focal_weight * self.focal(inputs, targets)
            + self.dice_weight * self.dice(inputs, targets)
        )
