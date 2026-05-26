from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from src.data.dataset import ISICDataset


def _create_fake_pair(
    img_dir,
    mask_dir,
    stem: str,
    *,
    segmentation_suffix: bool = False,
    h: int = 64,
    w: int = 64,
) -> None:
    """Create a fake image + mask pair on disk."""
    img = Image.fromarray(np.random.randint(0, 255, (h, w, 3), dtype=np.uint8))
    img.save(img_dir / f"{stem}.jpg")

    mask = Image.fromarray(np.zeros((h, w), dtype=np.uint8))
    mask_name = f"{stem}_segmentation.png" if segmentation_suffix else f"{stem}.png"
    mask.save(mask_dir / mask_name)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestISICDataset:
    def test_init_with_exact_stem_match(self, tmp_path) -> None:
        img_dir = tmp_path / "images"
        mask_dir = tmp_path / "masks"
        img_dir.mkdir()
        mask_dir.mkdir()
        _create_fake_pair(img_dir, mask_dir, "ISIC_001")
        _create_fake_pair(img_dir, mask_dir, "ISIC_002")

        ds = ISICDataset(img_dir, mask_dir)
        assert len(ds) == 2

    def test_init_with_segmentation_suffix(self, tmp_path) -> None:
        img_dir = tmp_path / "images"
        mask_dir = tmp_path / "masks"
        img_dir.mkdir()
        mask_dir.mkdir()
        _create_fake_pair(img_dir, mask_dir, "ISIC_001", segmentation_suffix=True)

        ds = ISICDataset(img_dir, mask_dir)
        assert len(ds) == 1

    def test_getitem_returns_correct_shapes(self, tmp_path) -> None:
        img_dir = tmp_path / "images"
        mask_dir = tmp_path / "masks"
        img_dir.mkdir()
        mask_dir.mkdir()
        _create_fake_pair(img_dir, mask_dir, "ISIC_001", h=32, w=48)

        ds = ISICDataset(img_dir, mask_dir)
        image, mask = ds[0]

        assert isinstance(image, np.ndarray)
        assert image.shape == (32, 48, 3)
        assert isinstance(mask, np.ndarray)
        assert mask.shape == (32, 48)
        assert set(np.unique(mask)).issubset({0.0, 1.0})

    def test_getitem_with_transform(self, tmp_path) -> None:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2

        img_dir = tmp_path / "images"
        mask_dir = tmp_path / "masks"
        img_dir.mkdir()
        mask_dir.mkdir()
        _create_fake_pair(img_dir, mask_dir, "ISIC_001", h=64, w=64)

        transform = A.Compose([A.Resize(32, 32), ToTensorV2()])
        ds = ISICDataset(img_dir, mask_dir, transform=transform)
        image, mask = ds[0]

        assert image.shape == (3, 32, 32)
        assert mask.shape == (1, 32, 32)

    def test_missing_img_dir_raises(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError, match="Image directory"):
            ISICDataset(tmp_path / "no_such_dir", tmp_path)

    def test_missing_mask_dir_raises(self, tmp_path) -> None:
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        with pytest.raises(FileNotFoundError, match="Mask directory"):
            ISICDataset(img_dir, tmp_path / "no_masks")

    def test_no_pairs_raises(self, tmp_path) -> None:
        img_dir = tmp_path / "images"
        mask_dir = tmp_path / "masks"
        img_dir.mkdir()
        mask_dir.mkdir()
        Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(img_dir / "a.jpg")

        with pytest.raises(ValueError, match="No image/mask pairs"):
            ISICDataset(img_dir, mask_dir)

    def test_repr(self, tmp_path) -> None:
        img_dir = tmp_path / "images"
        mask_dir = tmp_path / "masks"
        img_dir.mkdir()
        mask_dir.mkdir()
        _create_fake_pair(img_dir, mask_dir, "ISIC_001")

        ds = ISICDataset(img_dir, mask_dir)
        r = repr(ds)
        assert "n_samples=1" in r
