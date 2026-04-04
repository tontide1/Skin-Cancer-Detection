from __future__ import annotations

import numpy as np
import torch
from PIL import Image

import scripts.predict as predict_script
from scripts.evaluate import _resolve_eval_batch_size
from scripts.train import _resolve_find_unused_parameters
from src.utils.config import Config


def test_resolve_find_unused_parameters_defaults_to_false() -> None:
    cfg = Config({"training": {}})
    assert _resolve_find_unused_parameters(cfg) is False


def test_resolve_find_unused_parameters_reads_config_value() -> None:
    cfg = Config({"training": {"find_unused_parameters": True}})
    assert _resolve_find_unused_parameters(cfg) is True


def test_resolve_eval_batch_size_uses_multiplier_with_default_fallback() -> None:
    cfg_default = Config({"training": {"batch_size": 8}, "data": {}})
    cfg_custom = Config(
        {"training": {"batch_size": 8}, "data": {"val_batch_size_multiplier": 3}}
    )

    assert _resolve_eval_batch_size(cfg_default) == 16
    assert _resolve_eval_batch_size(cfg_custom) == 24


def test_preprocess_uses_provided_transform_without_rebuilding(monkeypatch, tmp_path) -> None:
    image_path = tmp_path / "sample.jpg"
    Image.fromarray(np.zeros((10, 12, 3), dtype=np.uint8)).save(image_path)

    monkeypatch.setattr(
        predict_script,
        "get_transforms",
        lambda *_args, **_kwargs: (
            _ for _ in ()
        ).throw(AssertionError("should not be called")),
    )

    calls = {"count": 0}

    def fake_transform(*, image, mask):
        calls["count"] += 1
        assert image.shape == (10, 12, 3)
        assert mask.shape == (10, 12)
        return {"image": torch.zeros((3, 8, 8), dtype=torch.float32)}

    out = predict_script.preprocess(image_path, fake_transform)

    assert calls["count"] == 1
    assert out.shape == (1, 3, 8, 8)
