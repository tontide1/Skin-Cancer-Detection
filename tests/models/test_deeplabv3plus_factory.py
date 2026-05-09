from __future__ import annotations

import os
import sys
import types

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import pytest
import torch

from src.models.segmentation import create_model
from src.utils.config import Config


class _FakeDeepLabV3Plus(torch.nn.Module):
    def __init__(self, classes: int):
        super().__init__()
        self.classes = classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        return torch.zeros((b, self.classes, h, w), dtype=x.dtype, device=x.device)


def _install_fake_smp(monkeypatch: pytest.MonkeyPatch, calls: list[dict]) -> None:
    fake_smp = types.ModuleType("segmentation_models_pytorch")

    def _fake_builder(**kwargs) -> _FakeDeepLabV3Plus:
        calls.append(kwargs.copy())
        return _FakeDeepLabV3Plus(classes=int(kwargs.get("classes", 1)))

    fake_smp.DeepLabV3Plus = _fake_builder
    monkeypatch.setitem(sys.modules, "segmentation_models_pytorch", fake_smp)


def _make_config(
    *,
    decoder_channels: int | list[int] = 256,
    encoder_output_stride: int = 16,
    decoder_attention_type: str | None = None,
) -> Config:
    return Config(
        {
            "model": {
                "name": "deeplabv3plus",
                "encoder_name": "resnet50",
                "encoder_weights": "imagenet",
                "in_channels": 3,
                "classes": 1,
                "decoder_channels": decoder_channels,
                "encoder_output_stride": encoder_output_stride,
                "decoder_attention_type": decoder_attention_type,
            }
        }
    )


def test_deeplabv3plus_factory_passes_expected_args(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict] = []
    _install_fake_smp(monkeypatch, calls)

    model = create_model(_make_config(decoder_channels=192, encoder_output_stride=8))
    y = model(torch.randn(2, 3, 64, 64))

    assert y.shape == (2, 1, 64, 64)
    assert len(calls) == 1
    assert calls[0]["encoder_name"] == "resnet50"
    assert calls[0]["encoder_weights"] == "imagenet"
    assert calls[0]["encoder_output_stride"] == 8
    assert calls[0]["decoder_channels"] == 192
    assert calls[0]["in_channels"] == 3
    assert calls[0]["classes"] == 1


def test_deeplabv3plus_factory_rejects_non_int_decoder_channels() -> None:
    with pytest.raises(ValueError, match="decoder_channels"):
        create_model(_make_config(decoder_channels=[256, 128]))


def test_deeplabv3plus_factory_rejects_invalid_output_stride() -> None:
    with pytest.raises(ValueError, match="encoder_output_stride"):
        create_model(_make_config(encoder_output_stride=4))


def test_deeplabv3plus_factory_warns_ignored_decoder_attention_type(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    calls: list[dict] = []
    _install_fake_smp(monkeypatch, calls)

    with caplog.at_level("WARNING", logger="src.models.segmentation"):
        create_model(_make_config(decoder_attention_type="scse"))

    assert len(calls) == 1
    assert any("decoder_attention_type" in record.message for record in caplog.records)
