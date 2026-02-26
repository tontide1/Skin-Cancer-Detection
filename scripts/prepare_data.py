#!/usr/bin/env python3
"""
Dataset preparation script — chia ISIC 2018 thành train/val/test (70/15/15).

Cấu trúc đầu vào (raw Kaggle download):
    data/raw/
        ISIC2018_Task1_Input/
            ISIC_0000000.jpg
            ...
        ISIC2018_Task1_GroundTruth/
            ISIC_0000000_segmentation.png
            ...

Cấu trúc đầu ra (data/processed):
    data/processed/
        train/
            images/  ISIC_0000000.jpg ...
            masks/   ISIC_0000000_segmentation.png ...
        val/
            images/
            masks/
        test/
            images/
            masks/

Usage:
    python scripts/prepare_data.py
    python scripts/prepare_data.py --raw-dir data/raw --out-dir data/processed --seed 42
"""

from __future__ import annotations

import argparse
import logging
import random
import shutil
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_valid_pairs(img_dir: Path, mask_dir: Path) -> list[str]:
    """Return sorted list of stems that have both image and mask files."""
    pairs = []
    for img_path in sorted(img_dir.glob("*.jpg")):
        stem = img_path.stem
        mask_path = mask_dir / f"{stem}_segmentation.png"
        if mask_path.exists():
            pairs.append(stem)
        else:
            log.warning(f"Missing mask for {stem} — skipped")
    return pairs


def _copy_split(
    stems: list[str],
    img_src: Path,
    mask_src: Path,
    img_dst: Path,
    mask_dst: Path,
    split_name: str,
) -> None:
    img_dst.mkdir(parents=True, exist_ok=True)
    mask_dst.mkdir(parents=True, exist_ok=True)

    for i, stem in enumerate(stems, 1):
        shutil.copy2(img_src  / f"{stem}.jpg",                img_dst  / f"{stem}.jpg")
        shutil.copy2(mask_src / f"{stem}_segmentation.png",   mask_dst / f"{stem}_segmentation.png")
        if i % 200 == 0:
            log.info(f"  {split_name}: {i}/{len(stems)} copied…")

    log.info(f"  {split_name}: {len(stems)} files done.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split ISIC 2018 dataset into train/val/test",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--raw-dir", default="data/raw",
                        help="Root of raw downloaded data (default: data/raw)")
    parser.add_argument("--out-dir", default="data/processed",
                        help="Output directory (default: data/processed)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio",   type=float, default=0.15)
    # test ratio = 1 - train - val
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)

    img_src  = raw_dir / "ISIC2018_Task1_Input"
    mask_src = raw_dir / "ISIC2018_Task1_GroundTruth"

    if not img_src.exists():
        log.error(f"Image dir not found: {img_src}")
        log.error("Expected structure: data/raw/ISIC2018_Task1_Input/*.jpg")
        sys.exit(1)
    if not mask_src.exists():
        log.error(f"Mask dir not found: {mask_src}")
        sys.exit(1)

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Get valid pairs
    stems = _get_valid_pairs(img_src, mask_src)
    log.info(f"Found {len(stems)} valid image-mask pairs")

    # Split
    test_ratio = 1.0 - args.train_ratio - args.val_ratio
    assert test_ratio > 0, "train_ratio + val_ratio must be < 1.0"

    train_stems, temp_stems = train_test_split(
        stems, test_size=(args.val_ratio + test_ratio),
        random_state=args.seed, shuffle=True,
    )
    val_stems, test_stems = train_test_split(
        temp_stems,
        test_size=test_ratio / (args.val_ratio + test_ratio),
        random_state=args.seed, shuffle=True,
    )

    total = len(stems)
    print("=" * 60)
    print("DATASET SPLIT")
    print("=" * 60)
    print(f"  Total:  {total}")
    print(f"  Train:  {len(train_stems)} ({len(train_stems)/total:.1%})")
    print(f"  Val:    {len(val_stems)}   ({len(val_stems)/total:.1%})")
    print(f"  Test:   {len(test_stems)}  ({len(test_stems)/total:.1%})")
    print("=" * 60)

    # Copy
    for split, s in [("train", train_stems), ("val", val_stems), ("test", test_stems)]:
        _copy_split(
            s,
            img_src,  mask_src,
            out_dir / split / "images",
            out_dir / split / "masks",
            split,
        )

    log.info(f"Done. Processed data at: {out_dir.resolve()}")
    log.info("Now run: python scripts/train.py --config configs/experiments/resnet34_unet_v1.yaml")


if __name__ == "__main__":
    main()
