from __future__ import annotations

import torch

from src.metrics.segmentation import dice_coefficient, find_best_threshold, iou_score


# ---------------------------------------------------------------------------
# dice_coefficient
# ---------------------------------------------------------------------------


class TestDiceCoefficient:
    def test_perfect_match(self) -> None:
        targets = torch.ones(2, 1, 16, 16)
        logits = torch.full_like(targets, 10.0)
        assert dice_coefficient(logits, targets) > 0.99

    def test_no_overlap(self) -> None:
        targets = torch.ones(2, 1, 16, 16)
        logits = torch.full_like(targets, -10.0)
        assert dice_coefficient(logits, targets) < 0.01

    def test_value_range(self) -> None:
        logits = torch.randn(4, 1, 16, 16)
        targets = torch.randint(0, 2, (4, 1, 16, 16)).float()
        d = dice_coefficient(logits, targets)
        assert 0.0 <= d <= 1.0

    def test_threshold_affects_result(self) -> None:
        logits = torch.zeros(2, 1, 16, 16)
        targets = torch.ones(2, 1, 16, 16)
        d_low = dice_coefficient(logits, targets, threshold=0.3)
        d_high = dice_coefficient(logits, targets, threshold=0.7)
        assert d_low > d_high

    def test_macro_averaging(self) -> None:
        """Each image contributes equally regardless of foreground area."""
        target1 = torch.ones(1, 1, 16, 16)
        target2 = torch.zeros(1, 1, 16, 16)
        logits = torch.full((2, 1, 16, 16), 10.0)
        targets = torch.cat([target1, target2], dim=0)

        d = dice_coefficient(logits, targets)
        assert 0.4 < d < 0.6


# ---------------------------------------------------------------------------
# iou_score
# ---------------------------------------------------------------------------


class TestIoUScore:
    def test_perfect_match(self) -> None:
        targets = torch.ones(2, 1, 16, 16)
        logits = torch.full_like(targets, 10.0)
        assert iou_score(logits, targets) > 0.99

    def test_no_overlap(self) -> None:
        targets = torch.ones(2, 1, 16, 16)
        logits = torch.full_like(targets, -10.0)
        assert iou_score(logits, targets) < 0.01

    def test_iou_less_than_dice(self) -> None:
        """IoU <= Dice for all cases with partial overlap."""
        logits = torch.randn(4, 1, 16, 16)
        targets = torch.randint(0, 2, (4, 1, 16, 16)).float()
        d = dice_coefficient(logits, targets)
        i = iou_score(logits, targets)
        assert i <= d + 1e-6


# ---------------------------------------------------------------------------
# find_best_threshold
# ---------------------------------------------------------------------------


class TestFindBestThreshold:
    def test_returns_tuple(self) -> None:
        logits = torch.randn(4, 1, 16, 16)
        targets = torch.randint(0, 2, (4, 1, 16, 16)).float()
        result = find_best_threshold(logits, targets)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_threshold_in_range(self) -> None:
        logits = torch.randn(4, 1, 16, 16)
        targets = torch.randint(0, 2, (4, 1, 16, 16)).float()
        thr, dice = find_best_threshold(logits, targets)
        assert 0.3 <= thr <= 0.7
        assert 0.0 <= dice <= 1.0

    def test_custom_thresholds(self) -> None:
        targets = torch.ones(2, 1, 8, 8)
        logits = torch.full_like(targets, 10.0)
        thr, dice = find_best_threshold(logits, targets, thresholds=[0.1, 0.5, 0.9])
        assert thr in {0.1, 0.5, 0.9}
        assert dice > 0.99
