"""
Utilities cho checkpoint loading với backward compatibility.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

_AUX_KEY_PREFIX = "aux_classifier."


def load_state_dict_with_aux_compat(
    model: torch.nn.Module,
    state_dict: dict[str, torch.Tensor],
    context: str = "checkpoint",
) -> None:
    """
    Load state_dict và bỏ qua legacy aux head keys nếu có.

    Hàm này hỗ trợ migration từ checkpoint DeepLab cũ (có aux head) sang
    model mới đã tắt aux head (`aux_loss=False`), nhưng vẫn giữ strict behavior
    cho mọi mismatch khác để tránh che giấu lỗi.

    Args:
        model: Model instance (có thể là DataParallel).
        state_dict: Trọng số model từ checkpoint.
        context: Context string để log/debug.

    Returns:
        None

    Raises:
        RuntimeError: Nếu có missing/unexpected keys ngoài aux head.
    """
    model_ref = model.module if hasattr(model, "module") else model
    incompatible = model_ref.load_state_dict(state_dict, strict=False)

    unexpected = list(incompatible.unexpected_keys)
    missing = list(incompatible.missing_keys)

    ignored_aux = [k for k in unexpected if k.startswith(_AUX_KEY_PREFIX)]
    other_unexpected = [k for k in unexpected if not k.startswith(_AUX_KEY_PREFIX)]
    other_missing = [k for k in missing if not k.startswith(_AUX_KEY_PREFIX)]

    if ignored_aux:
        logger.warning(
            "Ignored %d legacy aux key(s) while loading %s.",
            len(ignored_aux),
            context,
        )

    if other_missing or other_unexpected:
        problems: list[str] = []
        if other_missing:
            problems.append(f"missing keys: {other_missing}")
        if other_unexpected:
            problems.append(f"unexpected keys: {other_unexpected}")
        details = "; ".join(problems)
        raise RuntimeError(f"State dict mismatch while loading {context}: {details}")
