from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.models.segmentation import create_model
from src.models.transunet import TransUNet
from src.utils.config import Config


def _make_config(
    *,
    encoder_weights: str | None = None,
    pretrained_path: str | None = None,
    in_channels: int = 3,
    classes: int = 1,
    n_skip: int = 3,
) -> Config:
    return Config(
        {
            "model": {
                "name": "transunet",
                "transunet_variant": "r50_vit_b16",
                "encoder_name": "r50_vit_b16",
                "encoder_weights": encoder_weights,
                "pretrained_path": pretrained_path,
                "decoder_channels": [128, 64, 32, 16],
                "skip_channels": [512, 256, 64, 16],
                "n_skip": n_skip,
                "in_channels": in_channels,
                "classes": classes,
                "decoder_attention_type": None,
                "vit_hidden_size": 128,
                "vit_mlp_dim": 256,
                "vit_num_heads": 4,
                "vit_num_layers": 1,
                "vit_dropout_rate": 0.1,
                "vit_attention_dropout_rate": 0.0,
            },
            "data": {
                "input_size": [256, 256],
            },
        }
    )


def test_transunet_factory_returns_expected_class() -> None:
    model = create_model(_make_config())
    assert isinstance(model, TransUNet)


def test_transunet_factory_forward_shape_binary() -> None:
    model = create_model(_make_config())
    x = torch.randn(1, 3, 256, 256)
    y = model(x)
    assert y.shape == (1, 1, 256, 256)


def test_transunet_factory_rejects_non_rgb_input() -> None:
    with pytest.raises(ValueError, match="in_channels"):
        create_model(_make_config(in_channels=1))


def test_transunet_factory_rejects_non_binary_classes() -> None:
    with pytest.raises(ValueError, match="classes"):
        create_model(_make_config(classes=2))


def test_transunet_factory_rejects_invalid_n_skip() -> None:
    with pytest.raises(ValueError, match="n_skip"):
        create_model(_make_config(n_skip=4))


def test_transunet_factory_requires_pretrained_path_for_imagenet() -> None:
    with pytest.raises(ValueError, match="pretrained_path"):
        create_model(_make_config(encoder_weights="imagenet", pretrained_path=None))


def test_transunet_factory_calls_pretrained_loader(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    called: dict[str, str] = {}

    def _fake_load(self: TransUNet, checkpoint_path: str | Path) -> None:
        called["path"] = str(checkpoint_path)

    monkeypatch.setattr(TransUNet, "load_pretrained_from_npz", _fake_load)

    ckpt = tmp_path / "R50+ViT-B_16.npz"
    ckpt.write_bytes(b"dummy")

    _ = create_model(
        _make_config(encoder_weights="imagenet", pretrained_path=str(ckpt))
    )

    assert called.get("path") == str(ckpt)
