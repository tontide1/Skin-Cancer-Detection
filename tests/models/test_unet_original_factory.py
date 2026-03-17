from __future__ import annotations

import torch

from src.models.unet_original import UNetOriginal
from src.models.segmentation import create_model
from src.utils.config import Config


def _make_config(
    *,
    in_channels: int = 3,
    classes: int = 1,
    base_channels: int = 32,
) -> Config:
    return Config(
        {
            "model": {
                "name": "unet_original",
                "in_channels": in_channels,
                "classes": classes,
                "base_channels": base_channels,
                "encoder_name": None,
                "encoder_weights": None,
                "decoder_attention_type": None,
                "decoder_channels": None,
            }
        }
    )


def test_unet_original_factory_returns_expected_class() -> None:
    model = create_model(_make_config())
    assert isinstance(model, UNetOriginal)


def test_unet_original_factory_forward_shape_binary() -> None:
    model = create_model(_make_config(classes=1, base_channels=16))
    x = torch.randn(2, 3, 128, 128)
    y = model(x)
    assert y.shape == (2, 1, 128, 128)


def test_unet_original_factory_forward_shape_multiclass() -> None:
    model = create_model(_make_config(classes=3, base_channels=16))
    x = torch.randn(1, 3, 64, 64)
    y = model(x)
    assert y.shape == (1, 3, 64, 64)
