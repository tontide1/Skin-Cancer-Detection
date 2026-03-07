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
import torch.distributed as dist
import torch.nn as nn
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.losses import CombinedLoss
from src.metrics import dice_coefficient, iou_score
from src.training.callbacks import EarlyStopping, ModelCheckpoint
from src.utils.checkpoint import load_state_dict_with_aux_compat
from src.utils.logger import Logger
from src.utils.misc import plot_training_curves

logger = logging.getLogger(__name__)


def _compute_warmup_lr(
    epoch: int,
    warmup_epochs: int,
    warmup_start_lr: float,
    target_lr: float,
) -> float:
    """
    Compute linear warmup learning rate.

    Warmup progress uses (epoch + 1) / warmup_epochs so that the final warmup
    epoch reaches exactly target_lr.
    """
    if warmup_epochs <= 0:
        return target_lr
    progress = min((epoch + 1) / warmup_epochs, 1.0)
    return warmup_start_lr + (target_lr - warmup_start_lr) * progress


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
        is_distributed: bool = False,
        is_main_process: bool = True,
    ):
        self.model = model
        self.config = config
        self.device = device
        self.log = log
        self.is_distributed = is_distributed
        self.is_main_process = is_main_process
        self._resume_state: dict[str, Any] = {}

        # Loss
        self.criterion = CombinedLoss(config)

        # Optimizer
        self.optimizer = self._build_optimizer()

        # LR Scheduler
        self.scheduler = self._build_scheduler()

        # Mixed precision scaler (PyTorch 2.x API)
        self.scaler = GradScaler(device.type, enabled=config.training.mixed_precision)

        # Grad clipping
        self.grad_clip = config.training.grad_clip

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _build_optimizer(self) -> torch.optim.Optimizer:
        cfg = self.config.training
        name = cfg.optimizer.lower()

        params = self._get_param_groups(cfg)

        if name == "adamw":
            return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        if name == "adam":
            return torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        if name == "sgd":
            return torch.optim.SGD(
                params,
                lr=cfg.lr,
                weight_decay=cfg.weight_decay,
                momentum=0.9,
            )
        raise ValueError(f"Optimizer '{name}' chưa được hỗ trợ.")

    def _get_param_groups(self, cfg: Any) -> Any:
        """
        Build parameter groups for the optimizer.

        When ``training.differential_lr.enabled`` is True, splits parameters
        into encoder (lower LR) and decoder (full LR) groups.  Falls back
        gracefully to a single group if the model structure is not recognised.

        Supports:
        - SMP models: expose ``.encoder`` attribute
        - DeepLabV3Wrapper: expose ``.model`` attribute (torchvision)
        - DDP-wrapped models: unwrap ``.module`` first

        Args:
            cfg: ``config.training`` sub-config object.

        Returns:
            Either the model itself (single-LR path) or a list of param-group
            dicts (differential-LR path).
        """
        diff_lr_cfg = getattr(cfg, "differential_lr", None)
        if diff_lr_cfg is None or not getattr(diff_lr_cfg, "enabled", False):
            return self.model.parameters()

        # Unwrap DDP
        model_ref = self.model.module if hasattr(self.model, "module") else self.model

        encoder: torch.nn.Module | None = None

        # SMP models (UNet, DeepLabV3+ via SMP, ...)
        if hasattr(model_ref, "encoder"):
            encoder = model_ref.encoder
        # DeepLabV3Wrapper (torchvision) — backbone acts as encoder
        elif hasattr(model_ref, "model") and hasattr(model_ref.model, "backbone"):
            encoder = model_ref.model.backbone

        if encoder is None:
            logger.warning(
                "differential_lr.enabled=true but model exposes neither .encoder "
                "nor .model.backbone — falling back to single learning rate."
            )
            return self.model.parameters()

        encoder_lr = cfg.lr * float(getattr(diff_lr_cfg, "encoder_lr_scale", 0.1))
        encoder_ids = {id(p) for p in encoder.parameters()}

        decoder_params = [p for p in model_ref.parameters() if id(p) not in encoder_ids]
        encoder_params = list(encoder.parameters())

        logger.info(
            "Differential LR: encoder_lr=%.2e, decoder_lr=%.2e",
            encoder_lr,
            cfg.lr,
        )
        return [
            {"params": encoder_params, "lr": encoder_lr},
            {"params": decoder_params, "lr": cfg.lr},
        ]

    def _build_scheduler(self) -> Any:
        cfg = self.config.lr_schedule
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
            warmup_epochs = int(getattr(cfg, "warmup_epochs", 0) or 0)
            cosine_epochs = max(1, int(self.config.training.max_epochs) - warmup_epochs)
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=cosine_epochs,
                eta_min=cfg.min_lr,
            )
        raise ValueError(f"Scheduler '{name}' chưa được hỗ trợ.")

    def _sync_epoch_totals(
        self,
        total_loss: float,
        total_dice: float,
        total_iou: float,
        n_samples: int,
    ) -> tuple[float, float, float, int]:
        """
        Synchronize accumulated sums across DDP processes.
        """
        stats = torch.tensor(
            [total_loss, total_dice, total_iou, float(n_samples)],
            device=self.device,
            dtype=torch.float64,
        )
        if self.is_distributed:
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        return (
            float(stats[0].item()),
            float(stats[1].item()),
            float(stats[2].item()),
            int(stats[3].item()),
        )

    # ------------------------------------------------------------------
    # Core training
    # ------------------------------------------------------------------

    def train_one_epoch(self, loader: DataLoader) -> dict:
        """Train 1 epoch. Returns dict metrics."""
        self.model.train()
        total_loss = total_dice = total_iou = 0.0
        n_samples = 0

        pbar = tqdm(loader, desc="Train", leave=False, disable=not self.is_main_process)
        for images, masks in pbar:
            images = images.to(self.device)
            masks = masks.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                device_type=self.device.type,
                enabled=self.config.training.mixed_precision,
            ):
                logits = self.model(images)
                loss = self.criterion(logits, masks)

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
            total_iou += i * n
            n_samples += n

            if self.is_main_process:
                pbar.set_postfix(loss=f"{loss.item():.4f}", dice=f"{d:.4f}")

        total_loss, total_dice, total_iou, n_samples = self._sync_epoch_totals(
            total_loss, total_dice, total_iou, n_samples
        )
        if n_samples == 0:
            raise RuntimeError("Train loader không có sample nào.")
        return {
            "train_loss": total_loss / n_samples,
            "train_dice": total_dice / n_samples,
            "train_iou": total_iou / n_samples,
        }

    @torch.no_grad()
    def validate(self, loader: DataLoader) -> dict:
        """
        Validate model. Returns dict metrics.

        Dice và IoU được tính per-sample (accumulate intersection/union trực tiếp)
        thay vì per-batch-mean, tránh sai lệch do last incomplete batch.
        """
        self.model.eval()
        total_loss = 0.0
        # Per-sample accumulators (threshold=0.5)
        total_intersection = total_pred_sum = total_target_sum = 0.0
        n_samples = 0

        pbar = tqdm(loader, desc="Val  ", leave=False, disable=not self.is_main_process)
        for images, masks in pbar:
            images = images.to(self.device)
            masks = masks.to(self.device)

            with torch.amp.autocast(
                device_type=self.device.type,
                enabled=self.config.training.mixed_precision,
            ):
                logits = self.model(images)
                loss = self.criterion(logits, masks)

            n = images.size(0)
            total_loss += loss.item() * n

            # Per-sample Dice/IoU — accumulate raw counts, divide once at end
            probs = torch.sigmoid(logits)
            binary = (probs > 0.5).float()
            flat_pred = binary.view(n, -1)
            flat_mask = masks.float().view(n, -1)

            intersection = (flat_pred * flat_mask).sum(dim=1)
            pred_sum = flat_pred.sum(dim=1)
            target_sum = flat_mask.sum(dim=1)

            total_intersection += float(intersection.sum().item())
            total_pred_sum += float(pred_sum.sum().item())
            total_target_sum += float(target_sum.sum().item())
            n_samples += n

            if self.is_main_process:
                # Show approximate per-batch dice for progress bar
                batch_dice = float(
                    ((2.0 * intersection + 1e-7) / (pred_sum + target_sum + 1e-7)).mean().item()
                )
                pbar.set_postfix(dice=f"{batch_dice:.4f}")

        # Sync across DDP workers
        sync_stats = torch.tensor(
            [total_loss, total_intersection, total_pred_sum, total_target_sum, float(n_samples)],
            device=self.device,
            dtype=torch.float64,
        )
        if self.is_distributed:
            dist.all_reduce(sync_stats, op=dist.ReduceOp.SUM)
        total_loss = float(sync_stats[0].item())
        total_intersection = float(sync_stats[1].item())
        total_pred_sum = float(sync_stats[2].item())
        total_target_sum = float(sync_stats[3].item())
        n_samples = int(sync_stats[4].item())

        if n_samples == 0:
            raise RuntimeError("Validation loader không có sample nào.")

        eps = 1e-7
        val_dice = (2.0 * total_intersection + eps) / (total_pred_sum + total_target_sum + eps)
        val_iou = (total_intersection + eps) / (
            total_pred_sum + total_target_sum - total_intersection + eps
        )
        return {
            "val_loss": total_loss / n_samples,
            "val_dice": val_dice,
            "val_iou": val_iou,
        }

    # ------------------------------------------------------------------
    # Main fit loop
    # ------------------------------------------------------------------

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        output_dir: Path | str,
        start_epoch: int = 0,
        train_sampler: Any | None = None,
    ) -> dict:
        """
        Main training loop.

        Args:
            train_loader: DataLoader for training set
            val_loader:   DataLoader for validation set
            output_dir:   Nơi lưu best_model.pth và results
            start_epoch:  Epoch to resume from (0-indexed, default 0)
            train_sampler: Sampler (e.g., DistributedSampler) để gọi set_epoch()

        Returns:
            Dict tóm tắt kết quả training
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        cfg_es = self.config.early_stopping
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

        best_metrics: dict[str, Any] = {}
        if start_epoch > 0:
            prev_best, prev_best_epoch, prev_best_metrics = self._load_previous_best_state(
                output_dir, monitor
            )
            checkpoint.best = prev_best
            checkpoint.best_epoch = prev_best_epoch
            best_metrics = prev_best_metrics
            if self.is_main_process and prev_best is not None:
                epoch_text = (
                    f"epoch {prev_best_epoch + 1}"
                    if prev_best_epoch is not None
                    else "epoch unknown"
                )
                print(f"  Restored previous best {monitor}={prev_best:.4f} ({epoch_text})")

        lr_cfg = self.config.lr_schedule
        warmup_epochs = int(getattr(lr_cfg, "warmup_epochs", 0) or 0)
        warmup_start_lr = float(getattr(lr_cfg, "warmup_start_lr", 1e-6))
        target_lr = float(self.config.training.lr)

        if self.is_main_process:
            print("=" * 70)
            if start_epoch > 0:
                print(f"  TRAINING RESUMED from epoch {start_epoch + 1}")
            else:
                print("  TRAINING START")
            if warmup_epochs > 0:
                print(f"  Warmup: {warmup_epochs} epochs ({warmup_start_lr:.1e} → {target_lr:.1e})")
            print("=" * 70)

        last_epoch = start_epoch - 1
        for epoch in range(start_epoch, self.config.training.max_epochs):
            last_epoch = epoch
            if train_sampler is not None and hasattr(train_sampler, "set_epoch"):
                train_sampler.set_epoch(epoch)

            # --- Warmup LR (linear ramp before main scheduler) ---
            if epoch < warmup_epochs:
                warmup_lr = _compute_warmup_lr(
                    epoch=epoch,
                    warmup_epochs=warmup_epochs,
                    warmup_start_lr=warmup_start_lr,
                    target_lr=target_lr,
                )
                for pg in self.optimizer.param_groups:
                    pg["lr"] = warmup_lr

            # --- Train ---
            train_metrics = self.train_one_epoch(train_loader)
            val_metrics = self.validate(val_loader)

            lr = self.optimizer.param_groups[0]["lr"]
            epoch_metrics = {**train_metrics, **val_metrics, "lr": lr}

            # --- Log ---
            self.log.log(epoch_metrics, step=epoch + 1)

            # --- Print ---
            if self.is_main_process:
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
            is_best = False
            if self.is_main_process:
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

            # --- Save last checkpoint for resume ---
            if self.is_main_process:
                self._save_last_checkpoint(
                    output_dir=output_dir,
                    epoch=epoch,
                    checkpoint=checkpoint,
                    best_metrics=best_metrics,
                    monitor=monitor,
                )

            # --- LR Scheduler (skip during warmup) ---
            if epoch >= warmup_epochs:
                sched = self.config.lr_schedule.scheduler.lower()
                if sched == "reduce_on_plateau":
                    self.scheduler.step(val_metrics[monitor])
                else:
                    self.scheduler.step()

            # --- Early Stopping ---
            should_stop = False
            if self.is_main_process and early_stop.step(monitor_value):
                should_stop = True
                print(
                    f"\nEarly stopping triggered at epoch {epoch + 1}. "
                    f"Best {monitor}: {early_stop.best:.4f}"
                )

            if self.is_distributed:
                stop_tensor = torch.tensor(
                    int(should_stop),
                    device=self.device,
                    dtype=torch.int32,
                )
                dist.broadcast(stop_tensor, src=0)
                should_stop = bool(stop_tensor.item())

            if should_stop:
                break

        total_epochs = max(last_epoch + 1, start_epoch)
        summary = {
            "best_epoch": checkpoint.best_epoch,
            "best_metrics": best_metrics,
            "total_epochs": total_epochs,
        }
        if self.is_main_process:
            print("=" * 70)
            print("  TRAINING COMPLETE")
            print("=" * 70)

            # --- Save training curves ---
            curves_path = output_dir / "training_curves.png"
            plot_training_curves(self.log.history, curves_path)

            # --- Save training summary ---
            with open(output_dir / "training_summary.json", "w") as f:
                json.dump(summary, f, indent=2)

            self.log.log_summary(best_metrics)
        return summary

    def _save_last_checkpoint(
        self,
        output_dir: Path,
        epoch: int,
        checkpoint: ModelCheckpoint,
        best_metrics: dict[str, Any],
        monitor: str,
    ) -> None:
        """Save full training state for resume (overwrites each epoch)."""
        model_ref = self.model.module if hasattr(self.model, "module") else self.model
        payload = {
            "epoch": epoch,
            "model_state_dict": model_ref.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "monitor": monitor,
            "best": checkpoint.best,
            "best_epoch": checkpoint.best_epoch,
            "best_metrics": best_metrics,
            "training_history": self.log.history,
        }
        torch.save(payload, output_dir / "last_checkpoint.pth")

    def _load_previous_best_state(
        self,
        output_dir: Path,
        monitor: str,
    ) -> tuple[float | None, int | None, dict[str, Any]]:
        """
        Restore previous best metric state for resume-safe checkpointing.
        """
        best = self._resume_state.get("best")
        best_epoch = self._resume_state.get("best_epoch")
        best_metrics = self._resume_state.get("best_metrics")
        if not isinstance(best_metrics, dict):
            best_metrics = {}
        else:
            best_metrics = best_metrics.copy()

        best_model_path = output_dir / "best_model.pth"
        if best_model_path.exists():
            try:
                ckpt = torch.load(best_model_path, map_location="cpu", weights_only=True)
                if best is None and monitor in ckpt:
                    best = float(ckpt[monitor])
                if best_epoch is None and "epoch" in ckpt:
                    best_epoch = int(ckpt["epoch"])
            except Exception as exc:  # pragma: no cover - defensive path
                logger.warning("Không đọc được best_model.pth để restore best state: %s", exc)

        summary_path = output_dir / "training_summary.json"
        if summary_path.exists():
            try:
                with open(summary_path, "r") as f:
                    summary = json.load(f)
                if not best_metrics and isinstance(summary.get("best_metrics"), dict):
                    best_metrics = summary["best_metrics"]
                if best_epoch is None and summary.get("best_epoch") is not None:
                    best_epoch = int(summary["best_epoch"])
            except Exception as exc:  # pragma: no cover - defensive path
                logger.warning(
                    "Không đọc được training_summary.json để restore best state: %s", exc
                )

        if best is None and isinstance(best_metrics.get(monitor), (float, int)):
            best = float(best_metrics[monitor])

        return best, best_epoch, best_metrics

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def load_checkpoint(self, path: Path | str, resume: bool = False) -> dict:
        """
        Load model from checkpoint file.

        Args:
            path:   Path to checkpoint (.pth).
            resume: If True, also restore optimizer, scheduler, and scaler state
                    for continuing training.  If False, only model weights are loaded.

        Returns:
            The raw checkpoint dict (caller can inspect ``ckpt["epoch"]`` etc.).
        """
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        state = ckpt.get("model_state_dict", ckpt)
        load_state_dict_with_aux_compat(self.model, state, context=str(path))

        if resume:
            if "optimizer_state_dict" in ckpt:
                self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "scheduler_state_dict" in ckpt:
                self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            if "scaler_state_dict" in ckpt:
                self.scaler.load_state_dict(ckpt["scaler_state_dict"])
            # Restore training history so curves show the full run
            if "training_history" in ckpt and isinstance(ckpt["training_history"], list):
                self.log.history = ckpt["training_history"]
            self._resume_state = {
                "best": ckpt.get("best"),
                "best_epoch": ckpt.get("best_epoch"),
                "best_metrics": ckpt.get("best_metrics", {}),
            }
            logger.info(
                "Resumed full training state (optimizer + scheduler + scaler) from %s at epoch %d",
                path,
                ckpt.get("epoch", -1) + 1,
            )
        else:
            logger.info(f"Loaded model weights from: {path}")

        return ckpt
