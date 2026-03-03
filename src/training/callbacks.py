"""EarlyStopping và ModelCheckpoint callbacks."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Dừng training khi metric không cải thiện sau `patience` epochs.

    Args:
        patience:   Số epochs chờ trước khi dừng
        min_delta:  Ngưỡng cải thiện tối thiểu (để tránh noise)
        mode:       "max" (Dice, IoU) | "min" (Loss)
        monitor:    Tên metric để hiển thị log
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        mode: str = "max",
        monitor: str = "val_dice",
    ):
        self.patience  = patience
        self.min_delta = min_delta
        self.mode      = mode
        self.monitor   = monitor
        self.counter   = 0
        self.best      = None
        self.triggered = False

    def step(self, value: float) -> bool:
        """
        Kiểm tra metric hiện tại.

        Returns:
            True nếu nên dừng training
        """
        if self.best is None:
            self.best = value
            return False

        improved = (
            value > self.best + self.min_delta
            if self.mode == "max"
            else value < self.best - self.min_delta
        )

        if improved:
            self.best    = value
            self.counter = 0
        else:
            self.counter += 1
            logger.debug(
                f"EarlyStopping: {self.monitor} không cải thiện "
                f"({self.counter}/{self.patience})"
            )
            if self.counter >= self.patience:
                self.triggered = True

        return self.triggered

    def reset(self) -> None:
        self.counter   = 0
        self.best      = None
        self.triggered = False


class ModelCheckpoint:
    """
    Lưu best model dựa trên metric.

    Args:
        save_path:  Path để lưu checkpoint (.pth)
        mode:       "max" | "min"
        monitor:    Tên metric
    """

    def __init__(
        self,
        save_path: Path | str,
        mode: str = "max",
        monitor: str = "val_dice",
    ):
        self.save_path  = Path(save_path)
        self.mode       = mode
        self.monitor    = monitor
        self.best       = None
        self.best_epoch: int | None = None

    def step(
        self,
        value: float,
        model: torch.nn.Module,
        epoch: int,
        extra: dict | None = None,
        model_config: dict[str, Any] | None = None,
    ) -> bool:
        """
        Kiểm tra và lưu model nếu tốt hơn.

        Returns:
            True nếu checkpoint được lưu
        """
        is_best = (
            self.best is None
            or (self.mode == "max" and value > self.best)
            or (self.mode == "min" and value < self.best)
        )

        if is_best:
            self.best = value
            self.best_epoch = epoch
            payload = {
                "epoch": epoch,
                "model_state_dict": (
                    model.module.state_dict()
                    if hasattr(model, "module")  # DataParallel
                    else model.state_dict()
                ),
                self.monitor: value,
            }
            if model_config is not None:
                payload["model_config"] = model_config
            if extra:
                payload.update(extra)

            self.save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(payload, self.save_path)
            logger.info(
                f"Checkpoint saved → {self.save_path} "
                f"({self.monitor}={value:.4f}, epoch={epoch + 1})"
            )

        return is_best
