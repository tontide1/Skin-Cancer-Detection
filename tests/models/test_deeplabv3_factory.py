from __future__ import annotations

import sys
import os
import types

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import pytest
import torch

from src.models.segmentation import create_model
from src.utils.config import Config


class _FakeDeepLab(torch.nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        b, _, h, w = x.shape
        out = torch.zeros((b, self.num_classes, h, w), dtype=x.dtype, device=x.device)
        return {"out": out}


def _install_fake_torchvision(monkeypatch: pytest.MonkeyPatch, calls: list[dict]) -> type:
    fake_torchvision = types.ModuleType("torchvision")
    fake_models = types.ModuleType("torchvision.models")
    fake_segmentation = types.ModuleType("torchvision.models.segmentation")

    class _FakeMobileNetWeights:
        IMAGENET1K_V1 = object()

    def _fake_builder(
        *,
        weights=None,
        weights_backbone=None,
        num_classes: int | None = None,
        aux_loss: bool | None = None,
        **kwargs,
    ) -> _FakeDeepLab:
        calls.append(
            {
                "weights": weights,
                "weights_backbone": weights_backbone,
                "num_classes": num_classes,
                "aux_loss": aux_loss,
                "kwargs": kwargs,
            }
        )
        return _FakeDeepLab(num_classes or 21)

    fake_models.MobileNet_V3_Large_Weights = _FakeMobileNetWeights
    fake_segmentation.deeplabv3_mobilenet_v3_large = _fake_builder
    fake_torchvision.models = fake_models

    monkeypatch.setitem(sys.modules, "torchvision", fake_torchvision)
    monkeypatch.setitem(sys.modules, "torchvision.models", fake_models)
    monkeypatch.setitem(sys.modules, "torchvision.models.segmentation", fake_segmentation)
    return _FakeMobileNetWeights


def _make_config(encoder_weights: str | None) -> Config:
    return Config(
        {
            "model": {
                "name": "deeplabv3",
                "encoder_weights": encoder_weights,
                "classes": 1,
            }
        }
    )


def test_deeplabv3_factory_uses_backbone_imagenet_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict] = []
    fake_weights = _install_fake_torchvision(monkeypatch, calls)
    model = create_model(_make_config("imagenet"))

    x = torch.randn(2, 3, 32, 32)
    y = model(x)

    assert y.shape == (2, 1, 32, 32)
    assert len(calls) == 1
    assert calls[0]["weights"] is None
    assert calls[0]["weights_backbone"] is fake_weights.IMAGENET1K_V1
    assert calls[0]["num_classes"] == 1
    assert calls[0]["aux_loss"] is False


def test_deeplabv3_factory_uses_scratch_backbone_when_encoder_weights_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict] = []
    _install_fake_torchvision(monkeypatch, calls)
    create_model(_make_config(None))

    assert len(calls) == 1
    assert calls[0]["weights"] is None
    assert calls[0]["weights_backbone"] is None
    assert calls[0]["aux_loss"] is False


def test_deeplabv3_factory_rejects_invalid_encoder_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict] = []
    _install_fake_torchvision(monkeypatch, calls)

    with pytest.raises(ValueError, match="encoder_weights"):
        create_model(_make_config("coco"))
