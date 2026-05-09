from __future__ import annotations

import pytest
import torch

from src.utils.checkpoint import load_state_dict_with_aux_compat


def test_load_state_dict_ignores_legacy_aux_keys() -> None:
    model = torch.nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, bias=True)

    target_state = {
        "weight": torch.full_like(model.weight, 2.0),
        "bias": torch.full_like(model.bias, -1.0),
        "aux_classifier.4.weight": torch.randn(1, 1, 1, 1),
        "aux_classifier.4.bias": torch.randn(1),
    }

    load_state_dict_with_aux_compat(model, target_state, context="unit-test")

    assert torch.allclose(model.weight, target_state["weight"])
    assert torch.allclose(model.bias, target_state["bias"])


def test_load_state_dict_still_raises_for_non_aux_unexpected_keys() -> None:
    model = torch.nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, bias=True)
    state = model.state_dict()
    state["classifier.weight"] = torch.randn(1)

    with pytest.raises(RuntimeError, match="unexpected keys"):
        load_state_dict_with_aux_compat(model, state, context="unit-test")


def test_load_state_dict_still_raises_for_missing_non_aux_keys() -> None:
    model = torch.nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, bias=True)
    state = {"weight": model.weight.detach().clone()}

    with pytest.raises(RuntimeError, match="missing keys"):
        load_state_dict_with_aux_compat(model, state, context="unit-test")
