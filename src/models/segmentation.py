"""
Model factory cho segmentation.
Thêm model mới: implement build_<name>() rồi register vào create_model().
"""

from __future__ import annotations

import torch.nn as nn


# =============================================================================
# Builders
# =============================================================================

def _build_unet(config) -> nn.Module:
    """U-Net với SMP encoder backbone."""
    import segmentation_models_pytorch as smp

    m = config.model
    return smp.Unet(
        encoder_name=m.encoder_name,
        encoder_weights=m.encoder_weights,
        decoder_attention_type=m.decoder_attention_type,
        decoder_channels=list(m.decoder_channels),
        in_channels=m.in_channels,
        classes=m.classes,
    )


# =============================================================================
# Factory
# =============================================================================

_REGISTRY = {
    "unet": _build_unet,
    # "sam2": _build_sam2,   ← thêm model mới ở đây
}


def create_model(config) -> nn.Module:
    """
    Tạo model từ config.

    Args:
        config: Config object với config.model.name

    Returns:
        nn.Module

    Raises:
        ValueError: Nếu model name chưa được register

    Example:
        model = create_model(config)                    # local
        model = nn.DataParallel(model).to(device)       # multi-GPU
    """
    name = config.model.name
    if name not in _REGISTRY:
        available = list(_REGISTRY.keys())
        raise ValueError(
            f"Model '{name}' chưa được register. "
            f"Available: {available}"
        )
    return _REGISTRY[name](config)
