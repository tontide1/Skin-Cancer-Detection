from __future__ import annotations

import json

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.training.trainer import Trainer, _compute_warmup_lr
from src.utils.config import Config


class _NoOpLogger:
    def __init__(self) -> None:
        self.history: list[dict] = []

    def log(self, metrics: dict, step: int | None = None) -> None:
        if step is not None:
            metrics = {"step": step, **metrics}
        self.history.append(metrics)

    def log_summary(self, summary: dict) -> None:
        _ = summary

    def finish(self) -> None:
        return


class _ConstantLogitModel(torch.nn.Module):
    def __init__(self, init_logit: float = -10.0) -> None:
        super().__init__()
        self.logit = torch.nn.Parameter(torch.tensor(float(init_logit)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.logit.view(1, 1, 1, 1).expand(x.size(0), 1, x.size(2), x.size(3))


def _make_config(
    *,
    scheduler: str = "cosine",
    max_epochs: int = 6,
    warmup_epochs: int = 0,
    lr: float = 1.0e-3,
) -> Config:
    return Config(
        {
            "model": {"name": "dummy"},
            "training": {
                "optimizer": "sgd",
                "lr": lr,
                "weight_decay": 0.0,
                "mixed_precision": False,
                "grad_clip": None,
                "max_epochs": max_epochs,
                "batch_size": 2,
                "loss": {
                    "focal_weight": 0.5,
                    "dice_weight": 0.5,
                    "focal_gamma": 2.0,
                    "focal_alpha": None,
                },
            },
            "lr_schedule": {
                "scheduler": scheduler,
                "monitor": "val_dice",
                "mode": "max",
                "factor": 0.5,
                "patience": 2,
                "min_lr": 1.0e-7,
                "warmup_epochs": warmup_epochs,
                "warmup_start_lr": 1.0e-6,
            },
            "early_stopping": {
                "patience": 10,
                "min_delta": 1.0e-4,
                "mode": "max",
                "monitor": "val_dice",
            },
        }
    )


def _build_dummy_loader() -> DataLoader:
    images = torch.randn(4, 3, 8, 8)
    masks = torch.ones(4, 1, 8, 8)
    dataset = TensorDataset(images, masks)
    return DataLoader(dataset, batch_size=2, shuffle=False)


def test_compute_warmup_lr_reaches_target_on_last_warmup_epoch() -> None:
    start_lr = 1.0e-6
    target_lr = 1.0e-3
    warmup_epochs = 4

    first_epoch_lr = _compute_warmup_lr(0, warmup_epochs, start_lr, target_lr)
    last_epoch_lr = _compute_warmup_lr(
        warmup_epochs - 1, warmup_epochs, start_lr, target_lr
    )

    assert first_epoch_lr == pytest.approx(start_lr + (target_lr - start_lr) / warmup_epochs)
    assert last_epoch_lr == pytest.approx(target_lr)


def test_cosine_scheduler_tmax_excludes_warmup_epochs() -> None:
    cfg = _make_config(scheduler="cosine", max_epochs=12, warmup_epochs=3)
    trainer = Trainer(
        model=_ConstantLogitModel(),
        config=cfg,
        device=torch.device("cpu"),
        log=_NoOpLogger(),
    )
    assert trainer.scheduler.T_max == 9


def test_resume_preserves_previous_best_checkpoint(tmp_path) -> None:
    output_dir = tmp_path / "run"
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = _make_config(
        scheduler="reduce_on_plateau",
        max_epochs=6,
        warmup_epochs=0,
        lr=1.0e-6,
    )
    model = _ConstantLogitModel(init_logit=-10.0)
    trainer = Trainer(
        model=model,
        config=cfg,
        device=torch.device("cpu"),
        log=_NoOpLogger(),
    )
    loader = _build_dummy_loader()

    best_model_path = output_dir / "best_model.pth"
    torch.save(
        {
            "epoch": 4,
            "model_state_dict": model.state_dict(),
            "val_dice": 0.9,
            "val_iou": 0.82,
        },
        best_model_path,
    )
    with open(output_dir / "training_summary.json", "w") as f:
        json.dump(
            {
                "best_epoch": 4,
                "best_metrics": {
                    "train_loss": 0.2,
                    "train_dice": 0.85,
                    "train_iou": 0.76,
                    "val_loss": 0.1,
                    "val_dice": 0.9,
                    "val_iou": 0.82,
                    "lr": 1.0e-3,
                },
                "total_epochs": 5,
            },
            f,
            indent=2,
        )

    summary = trainer.fit(
        train_loader=loader,
        val_loader=loader,
        output_dir=output_dir,
        start_epoch=5,
    )

    assert summary["best_epoch"] == 4
    assert summary["best_metrics"]["val_dice"] == pytest.approx(0.9)

    best_ckpt_after = torch.load(best_model_path, map_location="cpu", weights_only=False)
    assert best_ckpt_after["val_dice"] == pytest.approx(0.9)
