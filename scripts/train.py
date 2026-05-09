#!/usr/bin/env python3
"""
Training entry point.

Usage (local):
    python scripts/train.py --config configs/experiments/resnet34_unet_v1.yaml

Usage (Kaggle):
    ! python scripts/train.py \
        --config configs/experiments/resnet34_unet_v1.yaml \
        data.root=/kaggle/input/isic-2018 \
        output.dir=/kaggle/working \
        logging.use_wandb=false

CLI overrides use dot-notation:   key.subkey=value
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Make sure the repo root is in PYTHONPATH so `src` can be imported
# when running as `python scripts/train.py` (without pip install -e .)
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from src.data.dataset import ISICDataset
from src.data.transforms import get_transforms
from src.models.segmentation import create_model
from src.training.distributed import (
    DistributedContext,
    parse_torchrun_env,
    single_process_context,
)
from src.training.trainer import Trainer
from src.utils.config import load_config, override_config
from src.utils.logger import Logger
from src.utils.misc import count_parameters, get_device, set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _NoOpLogger:
    """No-op logger for non-main DDP ranks."""

    def __init__(self) -> None:
        self.history: list[dict[str, Any]] = []

    def log(self, metrics: dict[str, float], step: int | None = None) -> None:
        _ = metrics, step

    def log_summary(self, summary: dict[str, float]) -> None:
        _ = summary

    def finish(self) -> None:
        return


def _resolve_find_unused_parameters(config) -> bool:
    """Resolve DDP find_unused_parameters from config with safe default."""
    return bool(getattr(config.training, "find_unused_parameters", False))


def _init_runtime(device_mode: str) -> tuple[torch.device, DistributedContext]:
    """
    Initialize single-GPU or torchrun-based DDP runtime.
    """
    if device_mode == "single":
        return get_device(), single_process_context()

    if not torch.cuda.is_available():
        raise RuntimeError("device-mode=ddp yêu cầu CUDA (GPU).")

    ctx = parse_torchrun_env(os.environ)
    if not ctx.enabled:
        raise RuntimeError(
            "device-mode=ddp yêu cầu WORLD_SIZE > 1. Hãy launch với torchrun --nproc_per_node=2 ..."
        )

    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(ctx.local_rank)
    device = torch.device(f"cuda:{ctx.local_rank}")
    return device, ctx


def _cleanup_runtime(ctx: DistributedContext) -> None:
    if ctx.enabled and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def build_dataloaders(
    config,
    dist_ctx: DistributedContext,
) -> tuple[DataLoader, DataLoader, DistributedSampler | None]:
    """Xây dựng train/val DataLoader từ config."""
    root = Path(config.data.root)

    train_ds = ISICDataset(
        img_dir=root / "train" / "images",
        mask_dir=root / "train" / "masks",
        transform=get_transforms("train", config),
    )
    val_ds = ISICDataset(
        img_dir=root / "val" / "images",
        mask_dir=root / "val" / "masks",
        transform=get_transforms("val", config),
    )

    train_sampler: DistributedSampler | None = None
    val_sampler: DistributedSampler | None = None
    if dist_ctx.enabled:
        train_sampler = DistributedSampler(
            train_ds,
            num_replicas=dist_ctx.world_size,
            rank=dist_ctx.rank,
            shuffle=True,
            drop_last=True,
        )
        val_sampler = DistributedSampler(
            val_ds,
            num_replicas=dist_ctx.world_size,
            rank=dist_ctx.rank,
            shuffle=False,
            drop_last=False,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.training.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        persistent_workers=config.data.persistent_workers and config.data.num_workers > 0,
        drop_last=True,
    )
    val_batch_size = config.training.batch_size * int(
        getattr(config.data, "val_batch_size_multiplier", 2)
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=val_batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        persistent_workers=config.data.persistent_workers and config.data.num_workers > 0,
    )

    if dist_ctx.is_main_process:
        log.info(f"Train: {len(train_ds)} samples | Val: {len(val_ds)} samples")
    return train_loader, val_loader, train_sampler


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train skin lesion segmentation model",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--config",
        "-c",
        required=True,
        help="Path to experiment YAML config file.\nExample: configs/experiments/resnet34_unet_v1.yaml",
    )
    parser.add_argument(
        "--resume",
        "-r",
        default=None,
        help="Path to last_checkpoint.pth to resume training from.",
    )
    parser.add_argument(
        "--device-mode",
        choices=["single", "ddp"],
        default="single",
        help=(
            "single: local single-GPU/CPU training (default)\n"
            "ddp: torchrun-based multi-GPU training (Kaggle 2xT4)"
        ),
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        metavar="key.subkey=value",
        help="Config overrides in dot-notation. Example: data.root=/kaggle/input/isic",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dist_ctx = single_process_context()
    run_logger: Logger | _NoOpLogger = _NoOpLogger()

    try:
        # 1. Load & override config
        config = load_config(args.config)
        config = override_config(config, args.overrides)

        # Auto-set experiment name from config file stem if not provided
        if not config.logging.experiment_name:
            config["logging"]["experiment_name"] = Path(args.config).stem

        # 2. Reproducibility
        set_seed(config.seed, deterministic=bool(getattr(config.training, "deterministic", True)))

        # 3. Device / distributed runtime
        device, dist_ctx = _init_runtime(args.device_mode)
        if dist_ctx.enabled and not dist_ctx.is_main_process:
            logging.getLogger().setLevel(logging.WARNING)
        if dist_ctx.is_main_process:
            log.info(
                "Device: %s | mode=%s | world_size=%d",
                device,
                args.device_mode,
                dist_ctx.world_size,
            )

        # 4. Output directory
        output_dir = Path(config.output.dir) / config.logging.experiment_name
        output_dir.mkdir(parents=True, exist_ok=True)
        if dist_ctx.is_main_process:
            log.info(f"Output: {output_dir}")

        # 5. Logger (W&B + local) chỉ trên main process
        if dist_ctx.is_main_process:
            run_logger = Logger(config, output_dir)

        # 6. Data
        train_loader, val_loader, train_sampler = build_dataloaders(config, dist_ctx)

        # 7. Model
        model = create_model(config).to(device)
        if dist_ctx.enabled:
            model = DDP(
                model,
                device_ids=[dist_ctx.local_rank],
                output_device=dist_ctx.local_rank,
                find_unused_parameters=_resolve_find_unused_parameters(config),
            )

        model_ref = model.module if hasattr(model, "module") else model
        params = count_parameters(model_ref)
        if dist_ctx.is_main_process:
            log.info(
                f"Model: {config.model.name} | {config.model.encoder_name} | "
                f"params={params['trainable']:,} ({params['size_mb']:.1f} MB)"
            )

        # 8. Trainer
        trainer = Trainer(
            model=model,
            config=config,
            device=device,
            log=run_logger,
            is_distributed=dist_ctx.enabled,
            is_main_process=dist_ctx.is_main_process,
        )

        # 8b. Resume from checkpoint if requested
        start_epoch = 0
        if args.resume:
            ckpt = trainer.load_checkpoint(args.resume, resume=True)
            start_epoch = ckpt.get("epoch", -1) + 1
            if dist_ctx.is_main_process:
                log.info(f"Resuming from epoch {start_epoch + 1}")

        # 9. Fit
        summary = trainer.fit(
            train_loader,
            val_loader,
            output_dir,
            start_epoch=start_epoch,
            train_sampler=train_sampler,
        )
        if dist_ctx.enabled:
            dist.barrier()

        # 10. Finish logging (main process only)
        if dist_ctx.is_main_process:
            run_logger.finish()

            best_dice = summary["best_metrics"].get("val_dice")
            best_epoch = summary.get("best_epoch")
            if best_dice is not None and best_epoch is not None:
                log.info(f"Done. Best val_dice={best_dice:.4f} at epoch {best_epoch + 1}")
            elif best_dice is not None:
                log.info(f"Done. Best val_dice={best_dice:.4f}")
            else:
                log.warning("Training finished without improvement.")
    finally:
        _cleanup_runtime(dist_ctx)


if __name__ == "__main__":
    main()
