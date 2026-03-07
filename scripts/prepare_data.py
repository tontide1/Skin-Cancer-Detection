#!/usr/bin/env python3
"""
Dataset preparation script — sử dụng split chính thức của ISIC 2018 Task 1.

Cấu trúc đầu vào (thư mục data/ sau khi download từ ISIC):
    data/
        ISIC2018_Task1-2_Training_Input/       ← 2594 ảnh .jpg
        ISIC2018_Task1_Training_GroundTruth/   ← 2594 masks _segmentation.png
        ISIC2018_Task1-2_Validation_Input/     ← 100 ảnh .jpg
        ISIC2018_Task1_Validation_GroundTruth/ ← 100 masks _segmentation.png
        ISIC2018_Task1-2_Test_Input/           ← 1000 ảnh .jpg
        ISIC2018_Task1_Test_GroundTruth/       ← 1000 masks _segmentation.png

Cấu trúc đầu ra (data/processed):
    data/processed/
        train/
            images/   ISIC_xxxxxxx.jpg ...
            masks/    ISIC_xxxxxxx_segmentation.png ...
        val/
            images/
            masks/
        test/
            images/
            masks/

Split chính thức ISIC 2018 Task 1:
    Train : 2594 ảnh  (~70.2%)
    Val   :  100 ảnh  (~ 2.7%)
    Test  : 1000 ảnh  (~27.1%)

Usage:
    python scripts/prepare_data.py
    python scripts/prepare_data.py --data-dir data/ --out-dir data/processed
"""

from __future__ import annotations

import argparse
import logging
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

# Mapping: split name → (images subdir, masks subdir)
_SPLIT_DIRS: dict[str, tuple[str, str]] = {
    "train": (
        "ISIC2018_Task1-2_Training_Input",
        "ISIC2018_Task1_Training_GroundTruth",
    ),
    "val": (
        "ISIC2018_Task1-2_Validation_Input",
        "ISIC2018_Task1_Validation_GroundTruth",
    ),
    "test": (
        "ISIC2018_Task1-2_Test_Input",
        "ISIC2018_Task1_Test_GroundTruth",
    ),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_valid_pairs(img_dir: Path, mask_dir: Path) -> list[str]:
    """
    Return sorted list of stems that have both image (.jpg) and mask (_segmentation.png).

    Args:
        img_dir:  Directory containing ISIC_xxxxxxx.jpg files.
        mask_dir: Directory containing ISIC_xxxxxxx_segmentation.png files.
    Returns:
        Sorted list of image stems with valid pairs.
    """
    pairs = []
    for img_path in sorted(img_dir.glob("*.jpg")):
        stem = img_path.stem
        mask_path = mask_dir / f"{stem}_segmentation.png"
        if mask_path.exists():
            pairs.append(stem)
        else:
            log.warning(f"Không tìm thấy mask cho {stem} — bỏ qua")
    return pairs


def _copy_split(
    stems: list[str],
    img_src: Path,
    mask_src: Path,
    img_dst: Path,
    mask_dst: Path,
    split_name: str,
) -> None:
    """
    Copy image-mask pairs vào thư mục đích.

    Args:
        stems:      List of image stems to copy.
        img_src:    Source directory for images.
        mask_src:   Source directory for masks.
        img_dst:    Destination directory for images.
        mask_dst:   Destination directory for masks.
        split_name: Name of the split (for logging).
    """
    img_dst.mkdir(parents=True, exist_ok=True)
    mask_dst.mkdir(parents=True, exist_ok=True)

    for i, stem in enumerate(stems, 1):
        shutil.copy2(img_src / f"{stem}.jpg", img_dst / f"{stem}.jpg")
        shutil.copy2(
            mask_src / f"{stem}_segmentation.png", mask_dst / f"{stem}_segmentation.png"
        )
        if i % 200 == 0:
            log.info(f"  {split_name}: {i}/{len(stems)} copied…")

    log.info(f"  {split_name}: {len(stems)} files done.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Chuẩn bị dataset ISIC 2018 Task 1 theo split chính thức.\n"
            "Train: 2594 | Val: 100 | Test: 1000"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        default="data/",
        help=("Thư mục chứa các folder ISIC 2018 gốc\n(default: data/)"),
    )
    parser.add_argument(
        "--out-dir",
        default="data/processed",
        help="Thư mục đầu ra (default: data/processed)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)

    # Validate tất cả thư mục nguồn tồn tại trước khi bắt đầu copy
    for split_name, (img_subdir, mask_subdir) in _SPLIT_DIRS.items():
        img_dir = data_dir / img_subdir
        mask_dir = data_dir / mask_subdir
        if not img_dir.exists():
            log.error(f"Không tìm thấy thư mục ảnh [{split_name}]: {img_dir}")
            sys.exit(1)
        if not mask_dir.exists():
            log.error(f"Không tìm thấy thư mục mask [{split_name}]: {mask_dir}")
            sys.exit(1)

    print("=" * 60)
    print("ISIC 2018 TASK 1 — OFFICIAL SPLIT")
    print("=" * 60)

    total = 0
    split_counts: dict[str, int] = {}

    for split_name, (img_subdir, mask_subdir) in _SPLIT_DIRS.items():
        img_src = data_dir / img_subdir
        mask_src = data_dir / mask_subdir

        stems = _get_valid_pairs(img_src, mask_src)
        split_counts[split_name] = len(stems)
        total += len(stems)

        log.info(f"Processing [{split_name}]: {len(stems)} pairs từ {img_subdir}")
        _copy_split(
            stems,
            img_src,
            mask_src,
            out_dir / split_name / "images",
            out_dir / split_name / "masks",
            split_name,
        )

    grand_total = sum(split_counts.values())
    print("=" * 60)
    for split_name, count in split_counts.items():
        print(f"  {split_name:<6}: {count:>5}  ({count / grand_total:.1%})")
    print(f"  {'Total':<6}: {grand_total:>5}")
    print("=" * 60)

    log.info(f"Done. Processed data tại: {out_dir.resolve()}")
    log.info(
        "Chạy tiếp: python scripts/train.py --config configs/experiments/resnet34_unet_v1.yaml"
    )


if __name__ == "__main__":
    main()
