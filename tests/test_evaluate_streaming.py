from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from scripts.evaluate import evaluate
from src.metrics.segmentation import dice_coefficient, iou_score


class _IdentityLogitModel(torch.nn.Module):
    """Return first channel as logits to keep deterministic behavior."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :1]


def _reference_eval(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_logits: list[torch.Tensor] = []
    all_masks: list[torch.Tensor] = []

    with torch.no_grad():
        for images, masks in loader:
            logits = model(images)
            loss = criterion(logits, masks)
            n = images.size(0)
            total_loss += loss.item() * n
            total_samples += n
            all_logits.append(logits)
            all_masks.append(masks)

    logits_t = torch.cat(all_logits, dim=0)
    masks_t = torch.cat(all_masks, dim=0)
    thresholds = [round(t, 2) for t in torch.arange(0.3, 0.71, 0.05).tolist()]
    dice_by_thr = [dice_coefficient(logits_t, masks_t, thr) for thr in thresholds]
    iou_by_thr = [iou_score(logits_t, masks_t, thr) for thr in thresholds]
    best_idx = max(range(len(thresholds)), key=lambda i: dice_by_thr[i])
    idx_05 = thresholds.index(0.5)

    return {
        "loss": total_loss / total_samples,
        "dice": dice_by_thr[idx_05],
        "iou": iou_by_thr[idx_05],
        "best_threshold": thresholds[best_idx],
        "best_dice_at_best_thr": dice_by_thr[best_idx],
        "best_iou_at_best_thr": iou_by_thr[best_idx],
    }


def test_evaluate_streaming_matches_reference_no_tta() -> None:
    torch.manual_seed(7)
    images = torch.randn(6, 1, 8, 8)
    masks = torch.randint(0, 2, (6, 1, 8, 8)).float()
    loader = DataLoader(TensorDataset(images, masks), batch_size=2, shuffle=False)
    model = _IdentityLogitModel()
    criterion = torch.nn.BCEWithLogitsLoss()

    streaming = evaluate(
        model=model,
        loader=loader,
        device=torch.device("cpu"),
        criterion=criterion,  # type: ignore[arg-type]
        split="test",
        threshold=0.5,
        use_tta=False,
    )
    reference = _reference_eval(model, loader, criterion)

    assert streaming["loss"] == pytest.approx(reference["loss"], abs=1e-7)
    assert streaming["dice"] == pytest.approx(reference["dice"], abs=1e-7)
    assert streaming["iou"] == pytest.approx(reference["iou"], abs=1e-7)
    assert streaming["best_threshold"] == pytest.approx(reference["best_threshold"], abs=1e-12)
    assert streaming["best_dice_at_best_thr"] == pytest.approx(
        reference["best_dice_at_best_thr"], abs=1e-7
    )
    assert streaming["best_iou_at_best_thr"] == pytest.approx(
        reference["best_iou_at_best_thr"], abs=1e-7
    )
    assert streaming["test_dice"] == pytest.approx(reference["dice"], abs=1e-7)
    assert streaming["test_iou"] == pytest.approx(reference["iou"], abs=1e-7)
    assert streaming["test_dice_best"] == pytest.approx(
        reference["best_dice_at_best_thr"], abs=1e-7
    )
    assert streaming["test_iou_best"] == pytest.approx(reference["best_iou_at_best_thr"], abs=1e-7)


def test_evaluate_uses_split_specific_prefix_keys() -> None:
    torch.manual_seed(11)
    images = torch.randn(4, 1, 8, 8)
    masks = torch.randint(0, 2, (4, 1, 8, 8)).float()
    loader = DataLoader(TensorDataset(images, masks), batch_size=2, shuffle=False)
    model = _IdentityLogitModel()
    criterion = torch.nn.BCEWithLogitsLoss()

    results = evaluate(
        model=model,
        loader=loader,
        device=torch.device("cpu"),
        criterion=criterion,  # type: ignore[arg-type]
        split="val",
        use_tta=False,
    )

    assert "val_dice" in results
    assert "val_iou" in results
    assert "val_dice_best" in results
    assert "val_iou_best" in results
    assert "test_dice" not in results
    assert results["val_dice"] == pytest.approx(results["dice"], abs=1e-7)
    assert results["val_dice_best"] == pytest.approx(results["best_dice_at_best_thr"], abs=1e-7)
    assert results["val_iou_best"] == pytest.approx(results["best_iou_at_best_thr"], abs=1e-7)
