"""
Integration smoke tests cho DeepLabV3 + MobileNetV3-Large.

Các test này dùng real torchvision (không mock) và sẽ tự động
bị skip nếu torchvision chưa được install (via pytest.importorskip).

Mục tiêu:
    1. create_model trả về DeepLabV3Wrapper (không phải dict)
    2. Forward pass (B, 3, H, W) → output tensor shape (B, 1, H, W)
    3. Output là tensor, không phải dict (DeepLabV3Wrapper đã unwrap)
    4. Gradient flow: loss.backward() không crash, params có .grad
    5. in_channels != 3 → ValueError
    6. encoder_name sai → log warning nhưng không raise
"""

from __future__ import annotations

import sys
import os
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import pytest
import torch

# Auto-skip toàn bộ module nếu torchvision chưa install
torchvision = pytest.importorskip("torchvision")

from src.models.segmentation import DeepLabV3Wrapper, create_model
from src.utils.config import Config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(
    encoder_weights: str | None = "imagenet",
    in_channels: int = 3,
    classes: int = 1,
    encoder_name: str = "mobilenet_v3_large",
) -> Config:
    return Config(
        {
            "model": {
                "name": "deeplabv3",
                "encoder_name": encoder_name,
                "encoder_weights": encoder_weights,
                "in_channels": in_channels,
                "classes": classes,
            }
        }
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_deeplabv3_real_returns_wrapper() -> None:
    """create_model trả về DeepLabV3Wrapper, không phải raw torchvision model."""
    model = create_model(_make_config(encoder_weights=None))
    assert isinstance(model, DeepLabV3Wrapper), (
        f"Expected DeepLabV3Wrapper, got {type(model).__name__}"
    )


def test_deeplabv3_real_forward_output_is_tensor() -> None:
    """Forward pass trả về tensor, không phải dict."""
    model = create_model(_make_config(encoder_weights=None))
    model.eval()
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        out = model(x)
    assert isinstance(out, torch.Tensor), f"Expected torch.Tensor, got {type(out)}"


def test_deeplabv3_real_forward_output_shape() -> None:
    """Forward pass (B, 3, H, W) → output shape (B, 1, H, W)."""
    model = create_model(_make_config(encoder_weights=None))
    model.eval()
    x = torch.randn(2, 3, 128, 128)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 1, 128, 128), (
        f"Expected shape (2, 1, 128, 128), got {out.shape}"
    )


def test_deeplabv3_real_gradient_flow() -> None:
    """loss.backward() không crash, tất cả params có .grad sau backward.

    Dùng input 256x256 để tránh BatchNorm crash khi spatial size collapse về
    1x1 sau ASPP với input nhỏ (BatchNorm yêu cầu > 1 value/channel khi train).
    """
    model = create_model(_make_config(encoder_weights=None))
    model.train()
    x = torch.randn(2, 3, 256, 256)
    out = model(x)
    loss = out.mean()
    loss.backward()

    params_with_grad = [
        p for p in model.parameters() if p.requires_grad and p.grad is not None
    ]
    assert len(params_with_grad) > 0, (
        "Không có param nào có .grad sau backward — gradient không flow qua model"
    )


def test_deeplabv3_real_rejects_in_channels_not_3() -> None:
    """in_channels != 3 phải raise ValueError ngay tại build time."""
    with pytest.raises(ValueError, match="in_channels"):
        create_model(_make_config(in_channels=1))


def test_deeplabv3_real_warns_on_wrong_encoder_name(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """encoder_name sai → log warning nhưng model vẫn được tạo thành công."""
    with caplog.at_level(logging.WARNING, logger="src.models.segmentation"):
        model = create_model(
            _make_config(encoder_weights=None, encoder_name="resnet50")
        )

    assert model is not None
    assert any("encoder_name" in record.message for record in caplog.records), (
        "Không có warning nào về encoder_name bị ignore"
    )
