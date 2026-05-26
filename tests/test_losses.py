from __future__ import annotations

import pytest
import torch

from src.losses.segmentation import CombinedLoss, FocalLoss, SoftDiceLoss
from src.utils.config import Config


# ---------------------------------------------------------------------------
# FocalLoss
# ---------------------------------------------------------------------------


class TestFocalLoss:
    def test_output_scalar(self) -> None:
        loss_fn = FocalLoss(gamma=2.0)
        logits = torch.randn(2, 1, 16, 16)
        targets = torch.randint(0, 2, (2, 1, 16, 16)).float()
        loss = loss_fn(logits, targets)
        assert loss.ndim == 0

    def test_loss_non_negative(self) -> None:
        loss_fn = FocalLoss(gamma=2.0)
        logits = torch.randn(4, 1, 16, 16)
        targets = torch.randint(0, 2, (4, 1, 16, 16)).float()
        assert loss_fn(logits, targets).item() >= 0

    def test_gradient_flows(self) -> None:
        loss_fn = FocalLoss(gamma=2.0)
        logits = torch.randn(2, 1, 8, 8, requires_grad=True)
        targets = torch.ones(2, 1, 8, 8)
        loss = loss_fn(logits, targets)
        loss.backward()
        assert logits.grad is not None

    def test_alpha_weighting(self) -> None:
        logits = torch.randn(4, 1, 16, 16)
        targets = torch.randint(0, 2, (4, 1, 16, 16)).float()

        loss_no_alpha = FocalLoss(gamma=2.0, alpha=None)(logits, targets)
        loss_alpha = FocalLoss(gamma=2.0, alpha=0.75)(logits, targets)
        assert loss_no_alpha.item() != pytest.approx(loss_alpha.item(), abs=1e-6)


# ---------------------------------------------------------------------------
# SoftDiceLoss
# ---------------------------------------------------------------------------


class TestSoftDiceLoss:
    def test_perfect_prediction_near_zero(self) -> None:
        loss_fn = SoftDiceLoss()
        targets = torch.ones(2, 1, 16, 16)
        logits = torch.full_like(targets, 10.0)
        loss = loss_fn(logits, targets)
        assert loss.item() < 0.01

    def test_worst_prediction_near_one(self) -> None:
        loss_fn = SoftDiceLoss()
        targets = torch.ones(2, 1, 16, 16)
        logits = torch.full_like(targets, -10.0)
        loss = loss_fn(logits, targets)
        assert loss.item() > 0.95

    def test_gradient_flows(self) -> None:
        loss_fn = SoftDiceLoss()
        logits = torch.randn(2, 1, 8, 8, requires_grad=True)
        targets = torch.ones(2, 1, 8, 8)
        loss = loss_fn(logits, targets)
        loss.backward()
        assert logits.grad is not None


# ---------------------------------------------------------------------------
# CombinedLoss
# ---------------------------------------------------------------------------


class TestCombinedLoss:
    @staticmethod
    def _make_config(
        focal_weight: float = 0.5,
        dice_weight: float = 0.5,
    ) -> Config:
        return Config({
            "training": {
                "loss": {
                    "focal_weight": focal_weight,
                    "dice_weight": dice_weight,
                    "focal_gamma": 2.0,
                    "focal_alpha": None,
                }
            }
        })

    def test_output_scalar(self) -> None:
        cfg = self._make_config()
        loss_fn = CombinedLoss(cfg)
        logits = torch.randn(2, 1, 16, 16)
        targets = torch.randint(0, 2, (2, 1, 16, 16)).float()
        loss = loss_fn(logits, targets)
        assert loss.ndim == 0

    def test_weighted_sum(self) -> None:
        logits = torch.randn(2, 1, 16, 16)
        targets = torch.randint(0, 2, (2, 1, 16, 16)).float()

        focal_only = CombinedLoss(self._make_config(1.0, 0.0))(logits, targets)
        dice_only = CombinedLoss(self._make_config(0.0, 1.0))(logits, targets)
        combined = CombinedLoss(self._make_config(0.5, 0.5))(logits, targets)

        expected = 0.5 * focal_only + 0.5 * dice_only
        assert combined.item() == pytest.approx(expected.item(), abs=1e-5)
