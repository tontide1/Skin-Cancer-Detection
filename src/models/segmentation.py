"""
Model factory cho segmentation.
Thêm model mới: implement build_<name>() rồi register vào create_model().
"""

from __future__ import annotations

import logging
import torch
import torch.nn as nn

_log = logging.getLogger(__name__)


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


def _build_unet_original(config) -> nn.Module:
    """
    Original U-Net (plain encoder-decoder, không pretrained backbone).

    Supported config fields:
        - model.in_channels (int)
        - model.classes (int)
        - model.base_channels (optional int, default=64)

    Informational-only fields inherited from base config:
        - encoder_name
        - encoder_weights
        - decoder_attention_type
        - decoder_channels
    """
    from src.models.unet_original import UNetOriginal

    m = config.model
    base_channels = int(getattr(m, "base_channels", 64))

    ignored_fields = (
        "encoder_name",
        "encoder_weights",
        "decoder_attention_type",
        "decoder_channels",
    )
    for field in ignored_fields:
        value = getattr(m, field, None)
        if value is not None:
            _log.warning(
                "unet_original ignores model.%s=%r. "
                "Set to null in the experiment YAML to silence this warning.",
                field,
                value,
            )

    return UNetOriginal(
        in_channels=int(m.in_channels),
        num_classes=int(m.classes),
        base_channels=base_channels,
    )


def _build_deeplabv3(config) -> nn.Module:
    """
    DeepLabV3 với MobileNetV3-Large backbone (torchvision).

    Config mapping:
        encoder_weights: "imagenet" → backbone ImageNet pretrained
                         null      → train from scratch
        classes:         Số output classes (default 1 cho binary segmentation)

    Constraints:
        - in_channels phải là 3 (torchvision hard-codes RGB input)
        - encoder_name chỉ có giá trị informational; builder luôn dùng
          deeplabv3_mobilenet_v3_large bất kể giá trị được set
    """
    from torchvision.models import MobileNet_V3_Large_Weights
    from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large

    m = config.model

    # Guard: torchvision không expose in_channels cho deeplabv3_mobilenet_v3_large
    if m.in_channels != 3:
        raise ValueError(
            f"DeepLabV3 (torchvision) chỉ hỗ trợ in_channels=3 (RGB). "
            f"Nhận được: in_channels={m.in_channels}. "
            "Dùng SMP-based model nếu cần in_channels khác."
        )

    # Warn: encoder_name bị ignore vì builder này luôn dùng mobilenet_v3_large
    _EXPECTED_ENCODER = "mobilenet_v3_large"
    actual_encoder = getattr(m, "encoder_name", _EXPECTED_ENCODER)
    if actual_encoder != _EXPECTED_ENCODER:
        _log.warning(
            "_build_deeplabv3 chỉ build deeplabv3_mobilenet_v3_large. "
            "encoder_name=%r bị ignore (expected %r). "
            "Thêm builder mới nếu cần backbone khác.",
            actual_encoder,
            _EXPECTED_ENCODER,
        )

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


def _build_deeplabv3plus(config) -> nn.Module:
    """
    DeepLabV3+ với SMP encoder backbone.

    Supported config fields:
        - model.encoder_name
        - model.encoder_weights
        - model.in_channels
        - model.classes
        - model.decoder_channels (int, default=256)
        - model.encoder_output_stride (optional, 8 or 16)

    Informational-only fields inherited from base config:
        - decoder_attention_type
    """
    import segmentation_models_pytorch as smp

    m = config.model

    decoder_channels = getattr(m, "decoder_channels", 256)
    if not isinstance(decoder_channels, int):
        raise ValueError(
            "DeepLabV3+ yêu cầu model.decoder_channels là số nguyên (int). "
            f"Nhận được: {decoder_channels!r}"
        )
    if decoder_channels <= 0:
        raise ValueError(
            "DeepLabV3+ yêu cầu model.decoder_channels > 0. "
            f"Nhận được: {decoder_channels}"
        )

    output_stride = int(getattr(m, "encoder_output_stride", 16))
    if output_stride not in (8, 16):
        raise ValueError(
            "DeepLabV3+ chỉ hỗ trợ model.encoder_output_stride là 8 hoặc 16. "
            f"Nhận được: {output_stride}"
        )

    decoder_attention_type = getattr(m, "decoder_attention_type", None)
    if decoder_attention_type is not None:
        _log.warning(
            "deeplabv3plus (SMP) ignores model.decoder_attention_type=%r.",
            decoder_attention_type,
        )

    return smp.DeepLabV3Plus(
        encoder_name=m.encoder_name,
        encoder_weights=m.encoder_weights,
        encoder_output_stride=output_stride,
        decoder_channels=decoder_channels,
        in_channels=m.in_channels,
        classes=m.classes,
    )


# =============================================================================
# Factory
# =============================================================================

_REGISTRY = {
    "unet": _build_unet,
    "unet_original": _build_unet_original,
    "deeplabv3": _build_deeplabv3,
    "deeplabv3plus": _build_deeplabv3plus,
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
        raise ValueError(f"Model '{name}' chưa được register. Available: {available}")
    return _REGISTRY[name](config)
