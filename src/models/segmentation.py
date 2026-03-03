"""
Model factory cho segmentation.
Thêm model mới: implement build_<name>() rồi register vào create_model().
"""

from __future__ import annotations

import torch
import torch.nn as nn


# =============================================================================
# Wrappers
# =============================================================================

class DeepLabV3Wrapper(nn.Module):
    """
    Wrap torchvision DeepLabV3 output dict → raw logits tensor.

    torchvision trả về OrderedDict({'out': tensor, 'aux': tensor}),
    nhưng Trainer/losses/metrics expect tensor (B, C, H, W) trực tiếp.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)["out"]


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


def _build_deeplabv3(config) -> nn.Module:
    """
    DeepLabV3 với MobileNetV3-Large backbone (torchvision).

    Config mapping:
        encoder_weights: "imagenet" → backbone ImageNet pretrained
                         null      → train from scratch
        classes:         Số output classes (default 1 cho binary segmentation)
    """
    from torchvision.models import MobileNet_V3_Large_Weights
    from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large

    m = config.model

    # Official semantics:
    # - "imagenet": chỉ pretrained backbone
    # - None:       train from scratch
    encoder_weights = m.encoder_weights
    if isinstance(encoder_weights, str):
        encoder_weights = encoder_weights.lower()

    if encoder_weights == "imagenet":
        weights_backbone = MobileNet_V3_Large_Weights.IMAGENET1K_V1
    elif encoder_weights is None:
        weights_backbone = None
    else:
        raise ValueError(
            "DeepLabV3 chỉ hỗ trợ model.encoder_weights='imagenet' hoặc null. "
            f"Nhận được: {m.encoder_weights!r}"
        )

    model = deeplabv3_mobilenet_v3_large(
        weights=None,
        weights_backbone=weights_backbone,
        num_classes=m.classes,
        aux_loss=False,
    )

    return DeepLabV3Wrapper(model)


# =============================================================================
# Factory
# =============================================================================

_REGISTRY = {
    "unet": _build_unet,
    "deeplabv3": _build_deeplabv3,
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
