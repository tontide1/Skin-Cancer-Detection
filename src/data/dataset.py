from __future__ import annotations

"""
ISIC 2018 Task 1 – skin lesion segmentation dataset.

Expected directory layout (after running scripts/prepare_data.py):

    data/processed/
    ├── train/
    │   ├── images/   ← *.jpg  dermoscopy images
    │   └── masks/    ← *_segmentation.png  binary ground-truth masks
    ├── val/
    │   ├── images/
    │   └── masks/
    └── test/
        ├── images/
        └── masks/

Each mask pixel is 0 (background) or 255 (lesion).
The dataset returns the mask normalised to float32 ∈ {0.0, 1.0}.
"""

import logging
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

log = logging.getLogger(__name__)

# Supported image extensions (case-insensitive)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


class ISICDataset(Dataset):
    """
    PyTorch Dataset for the ISIC 2018 Challenge Task 1 segmentation data.

    The dataset reads RGB dermoscopy images and their corresponding binary
    segmentation masks.  A (Albumentations) transform can optionally be
    applied to both image and mask together.

    Args:
        img_dir:   Directory containing the input images.
        mask_dir:  Directory containing the binary ground-truth masks.
        transform: An Albumentations ``Compose`` object (or any callable that
                   accepts ``image`` and ``mask`` keyword arguments and returns
                   a dict with the same keys).  Pass ``None`` to skip
                   augmentation (raw numpy arrays are returned).

    Returns (per item):
        image: torch.Tensor of shape (3, H, W), dtype float32.
        mask:  torch.Tensor of shape (1, H, W), dtype float32, values ∈ {0, 1}.

    Raises:
        FileNotFoundError: If ``img_dir`` or ``mask_dir`` does not exist.
        ValueError:        If no matching image/mask pairs are found.
        FileNotFoundError: At item-load time if a mask file is missing.
    """

    def __init__(
        self,
        img_dir: str | Path,
        mask_dir: str | Path,
        transform=None,
    ) -> None:
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform

        # Validate directories exist
        if not self.img_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.img_dir}")
        if not self.mask_dir.exists():
            raise FileNotFoundError(f"Mask directory not found: {self.mask_dir}")

        # Collect all image paths and pair with their masks
        self.samples: list[tuple[Path, Path]] = self._collect_samples()

        if not self.samples:
            raise ValueError(
                f"No image/mask pairs found.\n"
                f"  img_dir:  {self.img_dir}\n"
                f"  mask_dir: {self.mask_dir}\n"
                f"Check that both directories are non-empty and extensions match."
            )

        log.debug(f"ISICDataset: {len(self.samples)} samples from {self.img_dir}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _collect_samples(self) -> list[tuple[Path, Path]]:
        """
        Build (image_path, mask_path) pairs.

        Matching strategy (in order of priority):
        1. Exact stem match:      image ``ISIC_0024306.jpg``  →  mask ``ISIC_0024306.png``
        2. Suffix ``_segmentation``: mask ``ISIC_0024306_segmentation.png``

        Both strategies handle the official ISIC folder layout as well as the
        normalised ``prepare_data.py`` output.
        """
        samples: list[tuple[Path, Path]] = []

        # Build a lookup: mask stem (lower-case) → mask path
        mask_lookup: dict[str, Path] = {}
        for mask_path in sorted(self.mask_dir.iterdir()):
            if mask_path.suffix.lower() in IMAGE_EXTS:
                mask_lookup[mask_path.stem.lower()] = mask_path

        for img_path in sorted(self.img_dir.iterdir()):
            if img_path.suffix.lower() not in IMAGE_EXTS:
                continue

            stem = img_path.stem.lower()

            # Strategy 1 – exact stem match
            mask_path = mask_lookup.get(stem)

            # Strategy 2 – ISIC official naming: <stem>_segmentation
            if mask_path is None:
                mask_path = mask_lookup.get(f"{stem}_segmentation")

            if mask_path is None:
                log.warning(f"No mask found for image: {img_path.name} — skipping.")
                continue

            samples.append((img_path, mask_path))

        return samples

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        """
        Load and return one (image, mask) pair.

        Returns:
            image: torch.Tensor (3, H, W) float32
            mask:  torch.Tensor (1, H, W) float32, values ∈ {0.0, 1.0}
        """
        img_path, mask_path = self.samples[index]

        # Load image as RGB numpy array (H, W, 3) uint8
        image = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)

        # Load mask as grayscale; binarise to float32 ∈ {0, 1}
        mask_raw = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)
        mask = (mask_raw > 127).astype(np.float32)  # (H, W)

        # Apply augmentations / preprocessing
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]  # tensor (3, H, W) after ToTensorV2
            mask = augmented["mask"]  # tensor (H, W)  after ToTensorV2

            # Ensure mask has channel dimension: (H, W) → (1, H, W)
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)

        return image, mask

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"ISICDataset("
            f"n_samples={len(self)}, "
            f"img_dir={self.img_dir}, "
            f"mask_dir={self.mask_dir}, "
            f"transform={'set' if self.transform else 'None'})"
        )
