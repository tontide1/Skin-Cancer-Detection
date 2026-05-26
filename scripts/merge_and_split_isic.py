from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed CLI namespace.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Merge ISIC train/val/test data, then randomly split into test_1/test_2/test_3 "
            "(each contains images/ and masks/)."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Path to dataset root containing train/, val/, test/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/kaggle/working/isic_2018_task1_resplit"),
        help="Output root path for test_1, test_2, test_3.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible split.",
    )
    parser.add_argument(
        "--copy-mode",
        type=str,
        choices=["copy", "move"],
        default="copy",
        help="copy: keep original data, move: move files to output.",
    )
    return parser.parse_args()


def _find_mask_for_image(mask_dir: Path, image_stem: str) -> Path | None:
    """Find corresponding mask for an image stem.

    Args:
        mask_dir: Directory containing mask files.
        image_stem: Stem of image file.

    Returns:
        Path to matching mask if found, else None.
    """
    candidates = [p for p in mask_dir.glob(f"{image_stem}.*") if p.is_file()]
    if not candidates:
        return None
    return sorted(candidates)[0]


def _collect_pairs(input_dir: Path) -> list[tuple[Path, Path]]:
    """Collect (image, mask) pairs from train/val/test.

    Args:
        input_dir: Dataset root.

    Returns:
        List of image-mask pairs.
    """
    pairs: list[tuple[Path, Path]] = []
    missing_masks: list[Path] = []

    for split_name in ("train", "val", "test"):
        image_dir = input_dir / split_name / "images"
        mask_dir = input_dir / split_name / "masks"

        if not image_dir.exists() or not mask_dir.exists():
            raise ValueError(
                f"Missing expected directory for split '{split_name}': {image_dir} or {mask_dir}"
            )

        image_files = sorted(
            [p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_EXTS]
        )

        for image_path in image_files:
            mask_path = _find_mask_for_image(mask_dir, image_path.stem)
            if mask_path is None:
                missing_masks.append(image_path)
                continue
            pairs.append((image_path, mask_path))

    if missing_masks:
        preview = "\n".join(str(p) for p in missing_masks[:10])
        raise ValueError(
            "Some images do not have matching masks. "
            f"Missing count: {len(missing_masks)}\nExample paths:\n{preview}"
        )

    if not pairs:
        raise ValueError("No image-mask pairs found.")

    return pairs


def _split_sizes(total: int, n_splits: int) -> list[int]:
    """Compute balanced split sizes.

    Args:
        total: Total number of items.
        n_splits: Number of splits.

    Returns:
        List of sizes with difference <= 1.
    """
    base = total // n_splits
    rem = total % n_splits
    return [base + (1 if i < rem else 0) for i in range(n_splits)]


def _safe_dest_path(dest_dir: Path, file_name: str) -> Path:
    """Create a non-colliding destination file path.

    Args:
        dest_dir: Destination directory.
        file_name: Original file name.

    Returns:
        A path that does not collide with existing files.
    """
    dest = dest_dir / file_name
    if not dest.exists():
        return dest

    stem = dest.stem
    suffix = dest.suffix
    idx = 1
    while True:
        candidate = dest_dir / f"{stem}__dup{idx}{suffix}"
        if not candidate.exists():
            return candidate
        idx += 1


def _transfer_file(src: Path, dst: Path, mode: str) -> None:
    """Transfer file by copy or move.

    Args:
        src: Source file path.
        dst: Destination file path.
        mode: Either 'copy' or 'move'.
    """
    if mode == "copy":
        shutil.copy2(src, dst)
    else:
        shutil.move(src, dst)


def run(input_dir: Path, output_dir: Path, seed: int, copy_mode: str) -> None:
    """Run merge + random split pipeline.

    Args:
        input_dir: Dataset root with train/val/test.
        output_dir: Output root for test_1/test_2/test_3.
        seed: Random seed.
        copy_mode: copy or move.
    """
    pairs = _collect_pairs(input_dir)
    rng = random.Random(seed)
    rng.shuffle(pairs)

    sizes = _split_sizes(len(pairs), 3)
    split_names = ["test_1", "test_2", "test_3"]

    for split_name in split_names:
        (output_dir / split_name / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split_name / "masks").mkdir(parents=True, exist_ok=True)

    start = 0
    for split_name, size in zip(split_names, sizes, strict=True):
        end = start + size
        split_pairs = pairs[start:end]
        start = end

        image_out = output_dir / split_name / "images"
        mask_out = output_dir / split_name / "masks"

        for img_src, mask_src in split_pairs:
            img_dst = _safe_dest_path(image_out, img_src.name)
            mask_dst = _safe_dest_path(mask_out, mask_src.name)
            _transfer_file(img_src, img_dst, copy_mode)
            _transfer_file(mask_src, mask_dst, copy_mode)

    print("Done splitting dataset")
    print(f"Input directory : {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Total pairs     : {len(pairs)}")
    print(f"Split sizes     : test_1={sizes[0]}, test_2={sizes[1]}, test_3={sizes[2]}")
    print(f"Mode            : {copy_mode}")


def main() -> None:
    """Entrypoint."""
    args = _parse_args()
    run(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        copy_mode=args.copy_mode,
    )


if __name__ == "__main__":
    main()
