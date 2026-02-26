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
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Make sure the repo root is in PYTHONPATH so `src` can be imported
# when running as `python scripts/train.py` (without pip install -e .)
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from torch.utils.data import DataLoader

from src.data.dataset import ISICDataset
from src.data.transforms import get_transforms
from src.models.segmentation import create_model
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

def build_dataloaders(config) -> tuple[DataLoader, DataLoader]:
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

    train_loader = DataLoader(
        train_ds,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        persistent_workers=config.data.persistent_workers and config.data.num_workers > 0,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        persistent_workers=config.data.persistent_workers and config.data.num_workers > 0,
    )

    log.info(f"Train: {len(train_ds)} samples | Val: {len(val_ds)} samples")
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train skin lesion segmentation model",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--config", "-c",
        required=True,
        help="Path to experiment YAML config file.\nExample: configs/experiments/resnet34_unet_v1.yaml",
    )
    # Catch-all for dot-notation overrides: key.subkey=value
    parser.add_argument(
        "overrides",
        nargs="*",
        metavar="key.subkey=value",
        help="Config overrides in dot-notation. Example: data.root=/kaggle/input/isic",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 1. Load & override config
    config = load_config(args.config)
    config = override_config(config, args.overrides)

    # Auto-set experiment name from config file stem if not provided
    if not config.logging.experiment_name:
        config["logging"]["experiment_name"] = Path(args.config).stem

    # 2. Reproducibility
    set_seed(config.seed)

    # 3. Device
    device = get_device()
    log.info(f"Device: {device}")

    # 4. Output directory
    output_dir = Path(config.output.dir) / config.logging.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Output: {output_dir}")

    # 5. Logger (W&B + local)
    logger = Logger(config, output_dir)

    # 6. Data
    train_loader, val_loader = build_dataloaders(config)

    # 7. Model
    model = create_model(config).to(device)
    params = count_parameters(model)
    log.info(
        f"Model: {config.model.name} | {config.model.encoder_name} | "
        f"params={params['trainable']:,} ({params['size_mb']:.1f} MB)"
    )

    # 8. Trainer
    trainer = Trainer(model, config, device, logger)

    # 9. Fit
    summary = trainer.fit(train_loader, val_loader, output_dir)

    # 10. Finish logging
    logger.finish()

    log.info(
        f"Done. Best val_dice={summary['best_metrics'].get('val_dice', 'N/A'):.4f} "
        f"at epoch {summary['best_epoch']}"
    )


if __name__ == "__main__":
    main()
