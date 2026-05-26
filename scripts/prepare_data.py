#!/usr/bin/env python3
"""
Dataset preparation script cho cau truc dataset gom 1 tap anh + 1 tap mask.

Input mac dinh:
    data/data-HA10000-remove-hair/
        remove-hair/images/   <- anh da lieu (vd: ISIC_0024306.jpg)
        masks/                <- mask tuong ung (vd: ISIC_0024306.png)

Output:
    data/processed/
        train/
            images/
            masks/
        val/
            images/
            masks/
        test/
            images/
            masks/

Pairing rule:
    image stem == mask stem, mask extension = .png
    vd: ISIC_0024306.jpg <-> ISIC_0024306.png

Usage:
    python scripts/prepare_data.py
    python scripts/prepare_data.py --data-dir data/data-HA10000-remove-hair --out-dir data/processed
    python scripts/prepare_data.py --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1 --seed 42
"""

from __future__ import annotations

import argparse
import logging
import random
import shutil
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _get_valid_pairs(img_dir: Path, mask_dir: Path) -> tuple[list[tuple[Path, Path]], int]:
    """
    Return image/mask pairs where mask name exactly matches image stem.

    Args:
        img_dir: Directory containing input images.
        mask_dir: Directory containing mask PNG files.
    Returns:
        Tuple of:
        - list of (image_path, mask_path)
        - number of skipped images due to missing mask
    """
    pairs: list[tuple[Path, Path]] = []
    skipped = 0

    for img_path in sorted(img_dir.iterdir()):
        if not img_path.is_file() or img_path.suffix.lower() not in _IMAGE_EXTS:
            continue

        stem = img_path.stem
        mask_path = mask_dir / f"{stem}.png"
        if mask_path.exists():
            pairs.append((img_path, mask_path))
        else:
            skipped += 1
            log.warning(f"Khong tim thay mask cho {img_path.name} -> bo qua")

    return pairs, skipped


def _split_pairs(
    pairs: list[tuple[Path, Path]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[tuple[Path, Path]]]:
    """
    Shuffle and split pairs into train/val/test.

    Args:
        pairs: All valid (image_path, mask_path) pairs.
        train_ratio: Train ratio.
        val_ratio: Validation ratio.
        test_ratio: Test ratio.
        seed: Random seed for reproducible split.
    Returns:
        Mapping split name -> list of pairs.
    """
    shuffled = pairs.copy()
    random.Random(seed).shuffle(shuffled)

    n_total = len(shuffled)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val

    train_pairs = shuffled[:n_train]
    val_pairs = shuffled[n_train : n_train + n_val]
    test_pairs = shuffled[n_train + n_val :]

    assert len(test_pairs) == n_test

    return {
        "train": train_pairs,
        "val": val_pairs,
        "test": test_pairs,
    }


def _copy_split(
    pairs: list[tuple[Path, Path]],
    img_dst: Path,
    mask_dst: Path,
    split_name: str,
) -> None:
    """
    Copy image/mask pairs into destination split directories.

    Args:
        pairs: List of (image_path, mask_path) to copy.
        img_dst: Destination image directory.
        mask_dst: Destination mask directory.
        split_name: Split name for logging.
    """
    img_dst.mkdir(parents=True, exist_ok=True)
    mask_dst.mkdir(parents=True, exist_ok=True)

    for i, (img_path, mask_path) in enumerate(pairs, 1):
        shutil.copy2(img_path, img_dst / img_path.name)
        shutil.copy2(mask_path, mask_dst / mask_path.name)
        if i % 200 == 0:
            log.info(f"  {split_name}: {i}/{len(pairs)} copied...")

    log.info(f"  {split_name}: {len(pairs)} files done.")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Chuan bi dataset segmentation tu 1 thu muc images + 1 thu muc masks.",
    )
    parser.add_argument(
        "--data-dir",
        default="data/data-HA10000-remove-hair",
        help="Root thu muc dataset (default: data/data-HA10000-remove-hair)",
    )
    parser.add_argument(
        "--images-subdir",
        default="remove-hair/images",
        help="Subdir images tinh tu --data-dir (default: remove-hair/images)",
    )
    parser.add_argument(
        "--masks-subdir",
        default="masks",
        help="Subdir masks tinh tu --data-dir (default: masks)",
    )
    parser.add_argument(
        "--out-dir",
        default="data/processed",
        help="Thu muc dau ra (default: data/processed)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Ti le train (default: 0.8)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Ti le val (default: 0.1)",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Ti le test (default: 0.1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed cho random split (default: 42)",
    )
    return parser.parse_args()


def _validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    """Validate split ratios."""
    ratios = {
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
    }
    for name, value in ratios.items():
        if value < 0:
            raise ValueError(f"{name} phai >= 0, nhan duoc {value}")

    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-8:
        raise ValueError(
            f"Tong train_ratio + val_ratio + test_ratio phai bang 1.0 (hien tai: {total:.6f})"
        )


def main() -> None:
    """Entry point."""
    args = parse_args()

    data_dir = Path(args.data_dir)
    img_dir = data_dir / args.images_subdir
    mask_dir = data_dir / args.masks_subdir
    out_dir = Path(args.out_dir)

    try:
        _validate_ratios(args.train_ratio, args.val_ratio, args.test_ratio)
    except ValueError as exc:
        log.error(str(exc))
        sys.exit(1)

    if not img_dir.exists():
        log.error(f"Khong tim thay thu muc images: {img_dir}")
        sys.exit(1)
    if not mask_dir.exists():
        log.error(f"Khong tim thay thu muc masks: {mask_dir}")
        sys.exit(1)

    print("=" * 60)
    print("PREPARE DATASET FROM SINGLE IMAGES/MASKS SOURCE")
    print("=" * 60)
    log.info(f"Images dir: {img_dir}")
    log.info(f"Masks  dir: {mask_dir}")
    log.info(
        "Split ratio: train=%.2f val=%.2f test=%.2f (seed=%d)",
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.seed,
    )

    pairs, skipped = _get_valid_pairs(img_dir, mask_dir)
    if not pairs:
        log.error("Khong tim thay cap image-mask hop le nao.")
        sys.exit(1)

    split_map = _split_pairs(
        pairs,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    split_counts: dict[str, int] = {}
    for split_name, split_pairs in split_map.items():
        split_counts[split_name] = len(split_pairs)
        log.info(f"Processing [{split_name}]: {len(split_pairs)} pairs")
        _copy_split(
            split_pairs,
            out_dir / split_name / "images",
            out_dir / split_name / "masks",
            split_name,
        )

    grand_total = sum(split_counts.values())
    print("=" * 60)
    for split_name in ("train", "val", "test"):
        count = split_counts.get(split_name, 0)
        print(f"  {split_name:<6}: {count:>5}  ({count / grand_total:.1%})")
    print(f"  {'Total':<6}: {grand_total:>5}")
    print("=" * 60)

    log.info(f"Skipped images (missing mask): {skipped}")
    log.info(f"Done. Processed data tai: {out_dir.resolve()}")
    log.info(
        "Chay tiep: python scripts/train.py --config configs/experiments/resnet34_unet_v1.yaml"
    )


if __name__ == "__main__":
    main()
