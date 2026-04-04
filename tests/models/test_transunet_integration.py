from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.models.transunet import (
    TransUNet,
    TransUNetConfig,
    _ResNetConfig,
    _TransformerConfig,
)


def _tiny_transunet() -> TransUNet:
    cfg = TransUNetConfig(
        patches_grid=(4, 4),
        hidden_size=64,
        transformer=_TransformerConfig(
            mlp_dim=128,
            num_heads=4,
            num_layers=1,
            attention_dropout_rate=0.0,
            dropout_rate=0.1,
        ),
        classifier="seg",
        decoder_channels=(64, 32, 16, 8),
        skip_channels=(256, 128, 32, 16),
        n_skip=3,
        n_classes=1,
        resnet=_ResNetConfig(num_layers=(1, 1, 1), width_factor=0.5),
    )
    return TransUNet(config=cfg, img_size=64, in_channels=3, vis=False)


def test_transunet_tiny_forward_shape() -> None:
    model = _tiny_transunet()
    x = torch.randn(2, 3, 64, 64)
    y = model(x)
    assert y.shape == (2, 1, 64, 64)


def test_transunet_tiny_backward_has_gradients() -> None:
    model = _tiny_transunet()
    x = torch.randn(2, 3, 64, 64)
    y = model(x)
    loss = y.mean()
    loss.backward()

    params_with_grad = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert len(params_with_grad) > 0


def test_transunet_resize_position_embeddings() -> None:
    model = _tiny_transunet()
    posemb = torch.randn(1, 17, 64)  # class token + 4x4 tokens
    resized = model._resize_positional_embeddings(posemb, ntok_new=25)
    assert resized.shape == (1, 25, 64)


def test_transunet_load_pretrained_missing_path_raises(tmp_path: Path) -> None:
    model = _tiny_transunet()
    with pytest.raises(ValueError, match="not found"):
        model.load_pretrained_from_npz(tmp_path / "missing.npz")
