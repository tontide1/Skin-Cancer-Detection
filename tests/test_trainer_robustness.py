from __future__ import annotations

import json

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.metrics.segmentation import dice_coefficient, iou_score
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


class _IdentityLogitModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :1] * self.scale


class _EncoderDecoderModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = torch.nn.Conv2d(3, 4, kernel_size=1)
        self.decoder = torch.nn.Conv2d(4, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


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
                "differential_lr": {
                    "enabled": False,
                    "encoder_lr_scale": 0.1,
                },
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
    warmup_epochs = 4

    first_epoch_factor = _compute_warmup_lr(0, warmup_epochs)
    last_epoch_factor = _compute_warmup_lr(warmup_epochs - 1, warmup_epochs)

    assert first_epoch_factor == pytest.approx(1 / warmup_epochs)
    assert last_epoch_factor == pytest.approx(1.0)


def test_cosine_scheduler_tmax_excludes_warmup_epochs() -> None:
    cfg = _make_config(scheduler="cosine", max_epochs=12, warmup_epochs=3)
    trainer = Trainer(
        model=_ConstantLogitModel(),
        config=cfg,
        device=torch.device("cpu"),
        log=_NoOpLogger(),
    )
    assert trainer.scheduler.T_max == 9


def test_validate_uses_macro_average_metrics() -> None:
    model = _IdentityLogitModel()
    cfg = _make_config(max_epochs=1)
    trainer = Trainer(
        model=model,
        config=cfg,
        device=torch.device("cpu"),
        log=_NoOpLogger(),
    )

    images = torch.zeros(3, 3, 4, 4)
    logits = torch.full((3, 1, 4, 4), -10.0)
    logits[0, 0, 0, 0] = 10.0
    logits[2, 0, :, :] = 10.0
    images[:, 0] = logits[:, 0]

    masks = torch.zeros(3, 1, 4, 4)
    masks[0, 0, 0, 0] = 1.0
    masks[1, 0, :, :] = 1.0
    masks[2, 0, :, :] = 1.0

    loader = DataLoader(TensorDataset(images, masks), batch_size=2, shuffle=False)
    metrics = trainer.validate(loader)

    expected_dice = dice_coefficient(logits, masks)
    expected_iou = iou_score(logits, masks)
    assert metrics["val_dice"] == pytest.approx(expected_dice)
    assert metrics["val_iou"] == pytest.approx(expected_iou)


def test_warmup_preserves_differential_lr_ratio() -> None:
    cfg = _make_config(warmup_epochs=2, lr=1.0e-3)
    cfg.training.differential_lr.enabled = True
    cfg.training.differential_lr.encoder_lr_scale = 0.1
    trainer = Trainer(
        model=_EncoderDecoderModel(),
        config=cfg,
        device=torch.device("cpu"),
        log=_NoOpLogger(),
    )

    assert [pg["warmup_base_lr"] for pg in trainer.optimizer.param_groups] == pytest.approx(
        [1.0e-4, 1.0e-3]
    )
    assert [pg["warmup_start_lr"] for pg in trainer.optimizer.param_groups] == pytest.approx(
        [1.0e-7, 1.0e-6]
    )

    factor = _compute_warmup_lr(0, cfg.lr_schedule.warmup_epochs)
    for pg in trainer.optimizer.param_groups:
        pg["lr"] = pg["warmup_start_lr"] + (pg["warmup_base_lr"] - pg["warmup_start_lr"]) * factor

    encoder_lr, decoder_lr = [pg["lr"] for pg in trainer.optimizer.param_groups]
    assert decoder_lr == pytest.approx(5.005e-4)
    assert encoder_lr / decoder_lr == pytest.approx(0.1)


def test_resume_preserves_previous_best_checkpoint_for_matching_semantics(tmp_path) -> None:
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
    last_ckpt_path = output_dir / "last_checkpoint.pth"

    best_model_path = output_dir / "best_model.pth"
    torch.save(
        {
            "epoch": 4,
            "model_state_dict": model.state_dict(),
            "val_dice": 0.9,
            "val_iou": 0.82,
            "val_metric_semantics": "macro_per_sample_v1",
        },
        best_model_path,
    )
    torch.save(
        {
            "epoch": 4,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "scheduler_state_dict": trainer.scheduler.state_dict(),
            "scaler_state_dict": trainer.scaler.state_dict(),
            "best": 0.9,
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
            "training_history": [],
            "val_metric_semantics": "macro_per_sample_v1",
        },
        last_ckpt_path,
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

    trainer.load_checkpoint(last_ckpt_path, resume=True)
    summary = trainer.fit(
        train_loader=loader,
        val_loader=loader,
        output_dir=output_dir,
        start_epoch=5,
    )

    assert summary["best_epoch"] == 4
    assert summary["best_metrics"]["val_dice"] == pytest.approx(0.9)

    best_ckpt_after = torch.load(best_model_path, map_location="cpu", weights_only=True)
    assert best_ckpt_after["val_dice"] == pytest.approx(0.9)


def test_resume_legacy_checkpoint_resets_best_tracking(tmp_path) -> None:
    output_dir = tmp_path / "run"
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = _make_config(
        scheduler="reduce_on_plateau",
        max_epochs=4,
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

    legacy_ckpt_path = output_dir / "last_checkpoint.pth"
    torch.save(
        {
            "epoch": 2,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "scheduler_state_dict": trainer.scheduler.state_dict(),
            "scaler_state_dict": trainer.scaler.state_dict(),
            "best": 0.9,
            "best_epoch": 1,
            "best_metrics": {"val_dice": 0.9},
            "training_history": [],
        },
        legacy_ckpt_path,
    )
    torch.save(
        {
            "epoch": 1,
            "model_state_dict": model.state_dict(),
            "val_dice": 0.9,
            "val_iou": 0.82,
        },
        output_dir / "best_model.pth",
    )
    with open(output_dir / "training_summary.json", "w") as f:
        json.dump({"best_epoch": 1, "best_metrics": {"val_dice": 0.9}}, f, indent=2)

    trainer.load_checkpoint(legacy_ckpt_path, resume=True)
    summary = trainer.fit(
        train_loader=loader,
        val_loader=loader,
        output_dir=output_dir,
        start_epoch=3,
    )

    assert summary["best_epoch"] == 3
    assert summary["best_metrics"]["val_dice"] != pytest.approx(0.9)


def test_reduce_on_plateau_uses_lr_schedule_monitor(tmp_path) -> None:
    output_dir = tmp_path / "run"
    cfg = _make_config(scheduler="reduce_on_plateau", max_epochs=1)
    cfg.lr_schedule.monitor = "train_loss"
    trainer = Trainer(
        model=_ConstantLogitModel(),
        config=cfg,
        device=torch.device("cpu"),
        log=_NoOpLogger(),
    )
    loader = _build_dummy_loader()

    recorded: list[float] = []

    class _Recorder:
        def step(self, value: float) -> None:
            recorded.append(value)

        def state_dict(self) -> dict:
            return {}

    trainer.scheduler = _Recorder()
    trainer.fit(loader, loader, output_dir=output_dir)

    assert len(recorded) == 1
    assert recorded[0] == pytest.approx(trainer.log.history[0]["train_loss"])


def test_reduce_on_plateau_raises_for_missing_lr_monitor(tmp_path) -> None:
    output_dir = tmp_path / "run"
    cfg = _make_config(scheduler="reduce_on_plateau", max_epochs=1)
    cfg.lr_schedule.monitor = "missing_metric"
    trainer = Trainer(
        model=_ConstantLogitModel(),
        config=cfg,
        device=torch.device("cpu"),
        log=_NoOpLogger(),
    )
    loader = _build_dummy_loader()

    with pytest.raises(ValueError, match="lr_schedule.monitor"):
        trainer.fit(loader, loader, output_dir=output_dir)
