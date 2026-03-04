from __future__ import annotations

"""
Albumentations transform pipelines for training and validation/test.

Rules (from AGENTS.md §6 – Data Augmentation):
- Train augmentations live **only** here, in ``get_transforms("train", config)``.
- Val/test: Resize + Normalize + ToTensorV2 only — zero augmentation.
- ImageNet normalisation: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225].
- config.data.input_size = [H, W]  (e.g. [256, 256]).
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2

# ImageNet statistics (RGB)
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

# Recognised split aliases that receive zero augmentation
_VAL_SPLITS = {"val", "valid", "validation", "test", "predict"}


def get_transforms(split: str, config) -> A.Compose:
    """
    Return an Albumentations ``Compose`` pipeline for the requested split.

    Args:
        split:  One of ``"train"``, ``"val"``, ``"test"``, ``"predict"``, etc.
                Values present in ``_VAL_SPLITS`` receive the val/test pipeline
                (resize + normalise only, zero augmentation).
                Unrecognised splits fall through to the training pipeline —
                pass ``"train"`` explicitly to get full augmentation.
        config: Project config object (``src.utils.config.Config``).
                Must expose ``config.data.input_size`` as a sequence ``[H, W]``.

    Returns:
        ``albumentations.Compose`` instance configured for the split.
        The pipeline always returns a dict with keys ``"image"`` (torch
        float32 tensor, shape C×H×W) and ``"mask"`` (torch float32
        tensor, shape H×W).

    Raises:
        TypeError: If ``config.data.input_size`` is not a list/tuple of two ints.
    """
    input_size = config.data.input_size
    if len(input_size) != 2:
        raise TypeError(f"config.data.input_size must be [H, W], got: {input_size}")
    height, width = int(input_size[0]), int(input_size[1])

    # ------------------------------------------------------------------
    # Common tail shared by every split
    # ------------------------------------------------------------------
    _tail = [
        A.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ToTensorV2(),  # image: HWC uint8 → CHW float32 tensor
        # mask:  HW  float32 → HW float32 tensor (no new axis)
    ]

    normalized_split = split.lower()

    # ------------------------------------------------------------------
    # Validation / test — no augmentation
    # ------------------------------------------------------------------
    if normalized_split in _VAL_SPLITS:
        return A.Compose(
            [A.Resize(height, width)] + _tail,
        )

    # ------------------------------------------------------------------
    # Training — geometry + colour augmentations
    # ------------------------------------------------------------------
    return A.Compose(
        [
            # --- Spatial ---
            A.Resize(height, width),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Affine(
                scale=(0.9, 1.1),
                translate_percent={"x": (-0.0625, 0.0625), "y": (-0.0625, 0.0625)},
                rotate=(-15, 15),
                border_mode=0,  # constant (black) padding
                p=0.5,
            ),
            # --- Elastic / morphological ---
            A.ElasticTransform(alpha=120.0, sigma=8.0, p=0.2),
            A.GridDistortion(num_steps=5, distort_limit=0.1, p=0.2),
            # --- Colour / intensity (image only — mask unaffected) ---
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5,
            ),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=10,
                p=0.3,
            ),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.GaussNoise(p=0.2),
            # --- Regularisation ---
            A.CoarseDropout(
                num_holes_range=(1, 8),
                hole_height_range=(0.05, 0.1),
                hole_width_range=(0.05, 0.1),
                fill=0,
                fill_mask=0,
                p=0.2,
            ),
        ]
        + _tail,
    )
