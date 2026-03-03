"""
Trainer class — quản lý toàn bộ training loop.
Tách biệt hoàn toàn khỏi config paths và Kaggle-specific code.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.losses import CombinedLoss
from src.metrics import dice_coefficient, iou_score
from src.training.callbacks import EarlyStopping, ModelCheckpoint
from src.utils.logger import Logger
from src.utils.misc import plot_training_curves

logger = logging.getLogger(__name__)


class Trainer:
    """
    Trainer cho binary segmentation.

    Args:
        model:     nn.Module (đã move sang device)
        config:    Config object
        device:    torch.device
        log:       Logger instance (W&B + local)

    Usage:
        trainer = Trainer(model, config, device, log)
        trainer.fit(train_loader, val_loader, output_dir)
    """

    def __init__(
        self,
        model: nn.Module,
        config: Any,
        device: torch.device,
        log: Logger,
    ):
        self.model  = model
        self.config = config
        self.device = device
        self.log    = log

        # Loss
        self.criterion = CombinedLoss(config)

        # Optimizer
        self.optimizer = self._build_optimizer()

        # LR Scheduler
        self.scheduler = self._build_scheduler()

        # Mixed precision scaler (PyTorch 2.x API)
        self.scaler = GradScaler("cuda", enabled=config.training.mixed_precision)

        # Grad clipping
        self.grad_clip = config.training.grad_clip

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _build_optimizer(self) -> torch.optim.Optimizer:
        cfg  = self.config.training
        name = cfg.optimizer.lower()
        if name == "adamw":
            return torch.optim.AdamW(
                self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
            )
        if name == "adam":
            return torch.optim.Adam(
                self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
            )
        if name == "sgd":
            return torch.optim.SGD(
                self.model.parameters(), lr=cfg.lr,
                weight_decay=cfg.weight_decay, momentum=0.9,
            )
        raise ValueError(f"Optimizer '{name}' chưa được hỗ trợ.")

    def _build_scheduler(self) -> Any:
        cfg  = self.config.lr_schedule
        name = cfg.scheduler.lower()
        if name == "reduce_on_plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=cfg.mode,
                factor=cfg.factor,
                patience=cfg.patience,
                min_lr=cfg.min_lr,
            )
        if name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training.max_epochs,
                eta_min=cfg.min_lr,
            )
        raise ValueError(f"Scheduler '{name}' chưa được hỗ trợ.")

    # ------------------------------------------------------------------
    # Core training
    # ------------------------------------------------------------------

    def train_one_epoch(self, loader: DataLoader) -> dict:
        """Train 1 epoch. Returns dict metrics."""
        self.model.train()
        total_loss = total_dice = total_iou = 0.0

        pbar = tqdm(loader, desc="Train", leave=False)
        for images, masks in pbar:
            images = images.to(self.device)
            masks  = masks.to(self.device)

            self.optimizer.zero_grad()

            with torch.amp.autocast(
                device_type=self.device.type,
                enabled=self.config.training.mixed_precision,
            ):
                logits = self.model(images)
                loss   = self.criterion(logits, masks)

            self.scaler.scale(loss).backward()

            if self.grad_clip:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Metrics (detach để không ảnh hưởng autograd)
            with torch.no_grad():
                d = dice_coefficient(logits, masks)
                i = iou_score(logits, masks)

            n = images.size(0)
            total_loss += loss.item() * n
            total_dice += d * n
            total_iou  += i * n

            pbar.set_postfix(loss=f"{loss.item():.4f}", dice=f"{d:.4f}")

        n_samples = len(loader.dataset)
        return {
            "train_loss": total_loss / n_samples,
            "train_dice": total_dice / n_samples,
            "train_iou":  total_iou  / n_samples,
        }

    @torch.no_grad()
    def validate(self, loader: DataLoader) -> dict:
        """Validate model. Returns dict metrics."""
        self.model.eval()
        total_loss = total_dice = total_iou = 0.0

        pbar = tqdm(loader, desc="Val  ", leave=False)
        for images, masks in pbar:
            images = images.to(self.device)
            masks  = masks.to(self.device)

            with torch.amp.autocast(
                device_type=self.device.type,
                enabled=self.config.training.mixed_precision,
            ):
                logits = self.model(images)
                loss   = self.criterion(logits, masks)

            d = dice_coefficient(logits, masks)
            i = iou_score(logits, masks)

            n = images.size(0)
            total_loss += loss.item() * n
            total_dice += d * n
            total_iou  += i * n

            pbar.set_postfix(dice=f"{d:.4f}")

        n_samples = len(loader.dataset)
        return {
            "val_loss": total_loss / n_samples,
            "val_dice": total_dice / n_samples,
            "val_iou":  total_iou  / n_samples,
        }

    # ------------------------------------------------------------------
    # Main fit loop
    # ------------------------------------------------------------------

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        output_dir: Path | str,
    ) -> dict:
        """
        Main training loop.

        Args:
            train_loader: DataLoader for training set
            val_loader:   DataLoader for validation set
            output_dir:   Nơi lưu best_model.pth và results

        Returns:
            Dict tóm tắt kết quả training
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        cfg_es  = self.config.early_stopping
        monitor = cfg_es.monitor  # "val_dice"

        checkpoint = ModelCheckpoint(
            save_path=output_dir / "best_model.pth",
            mode=cfg_es.mode,
            monitor=monitor,
        )
        early_stop = EarlyStopping(
            patience=cfg_es.patience,
            min_delta=cfg_es.min_delta,
            mode=cfg_es.mode,
            monitor=monitor,
        )

        best_metrics: dict = {}

        print("=" * 70)
        print("  TRAINING START")
        print("=" * 70)

        for epoch in range(self.config.training.max_epochs):
            # --- Train ---
            train_metrics = self.train_one_epoch(train_loader)
            val_metrics   = self.validate(val_loader)

            lr = self.optimizer.param_groups[0]["lr"]
            epoch_metrics = {**train_metrics, **val_metrics, "lr": lr}

            # --- Log ---
            self.log.log(epoch_metrics, step=epoch + 1)

            # --- Print ---
            print(
                f"Epoch {epoch + 1:03d} | "
                f"loss={train_metrics['train_loss']:.4f} "
                f"dice={train_metrics['train_dice']:.4f} | "
                f"val_loss={val_metrics['val_loss']:.4f} "
                f"val_dice={val_metrics['val_dice']:.4f} | "
                f"lr={lr:.2e}"
            )

            # --- Checkpoint ---
            monitor_value = val_metrics[monitor]
            is_best = checkpoint.step(
                monitor_value,
                self.model,
                epoch,
                extra={"val_iou": val_metrics["val_iou"]},
                model_config=self.config.model.to_dict(),
            )
            if is_best:
                best_metrics = epoch_metrics.copy()
                print(f"  ✓ Best model saved ({monitor}={monitor_value:.4f})")

            # --- LR Scheduler ---
            sched = self.config.lr_schedule.scheduler.lower()
            if sched == "reduce_on_plateau":
                self.scheduler.step(val_metrics[monitor])
            else:
                self.scheduler.step()

            # --- Early Stopping ---
            if early_stop.step(monitor_value):
                print(
                    f"\nEarly stopping triggered at epoch {epoch + 1}. "
                    f"Best {monitor}: {early_stop.best:.4f}"
                )
                break

        print("=" * 70)
        print("  TRAINING COMPLETE")
        print("=" * 70)

        # --- Save training curves ---
        curves_path = output_dir / "training_curves.png"
        plot_training_curves(self.log.history, curves_path)

        # --- Save training summary ---
        summary = {
            "best_epoch": checkpoint.best,
            "best_metrics": best_metrics,
            "total_epochs": epoch + 1,
        }
        with open(output_dir / "training_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        self.log.log_summary(best_metrics)
        return summary

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def load_checkpoint(self, path: Path | str) -> dict:
        """Load model từ checkpoint file."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        if hasattr(self.model, "module"):
            self.model.module.load_state_dict(state)
        else:
            self.model.load_state_dict(state)
        logger.info(f"Loaded checkpoint: {path}")
        return ckpt
