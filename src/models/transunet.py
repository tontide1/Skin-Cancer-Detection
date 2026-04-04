from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair


@dataclass(frozen=True)
class _ResNetConfig:
    num_layers: tuple[int, int, int]
    width_factor: float


@dataclass(frozen=True)
class _TransformerConfig:
    mlp_dim: int
    num_heads: int
    num_layers: int
    attention_dropout_rate: float
    dropout_rate: float


@dataclass(frozen=True)
class TransUNetConfig:
    """Typed runtime config for TransUNet.

    Args:
        patches_grid: Patch grid for hybrid embeddings.
        hidden_size: Transformer embedding size.
        transformer: Transformer hyperparameters.
        classifier: Classifier mode (kept for checkpoint compatibility).
        decoder_channels: Decoder feature channels from coarse to fine.
        skip_channels: Skip feature channels.
        n_skip: Number of skip levels to use.
        n_classes: Number of segmentation output channels.
        resnet: Hybrid ResNetV2 settings.
    """

    patches_grid: tuple[int, int] | None
    hidden_size: int
    transformer: _TransformerConfig
    classifier: str
    decoder_channels: tuple[int, int, int, int]
    skip_channels: tuple[int, int, int, int]
    n_skip: int
    n_classes: int
    resnet: _ResNetConfig


def build_r50_vit_b16_config(
    *,
    n_classes: int,
    decoder_channels: tuple[int, int, int, int],
    n_skip: int,
    hidden_size: int = 768,
    mlp_dim: int = 3072,
    num_heads: int = 12,
    num_layers: int = 12,
    dropout_rate: float = 0.1,
    attention_dropout_rate: float = 0.0,
    skip_channels: tuple[int, int, int, int] = (512, 256, 64, 16),
) -> TransUNetConfig:
    """Build typed config for R50-ViT-B_16 TransUNet.

    Args:
        n_classes: Number of output segmentation channels.
        decoder_channels: Decoder channels tuple of length 4.
        n_skip: Number of skip features used by decoder (0..3).
        hidden_size: Transformer hidden size.
        mlp_dim: Transformer MLP hidden size.
        num_heads: Number of attention heads.
        num_layers: Number of transformer blocks.
        dropout_rate: Dropout after embeddings and MLP.
        attention_dropout_rate: Attention dropout rate.
        skip_channels: Skip feature channels tuple of length 4.

    Returns:
        A configured ``TransUNetConfig`` instance.
    """
    return TransUNetConfig(
        patches_grid=(16, 16),
        hidden_size=int(hidden_size),
        transformer=_TransformerConfig(
            mlp_dim=int(mlp_dim),
            num_heads=int(num_heads),
            num_layers=int(num_layers),
            attention_dropout_rate=float(attention_dropout_rate),
            dropout_rate=float(dropout_rate),
        ),
        classifier="seg",
        decoder_channels=tuple(int(v) for v in decoder_channels),
        skip_channels=tuple(int(v) for v in skip_channels),
        n_skip=int(n_skip),
        n_classes=int(n_classes),
        resnet=_ResNetConfig(num_layers=(3, 4, 9), width_factor=1.0),
    )


def _np_to_torch(weights: np.ndarray, conv: bool = False) -> torch.Tensor:
    """Convert numpy weights to torch tensor (optionally HWIO->OIHW)."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def _copy_param(param: nn.Parameter, value: torch.Tensor) -> None:
    """Copy tensor into parameter with dtype/device alignment."""
    param.copy_(value.to(device=param.device, dtype=param.dtype))


class StdConv2d(nn.Conv2d):
    """Weight-standardized Conv2d used by the hybrid ResNet backbone."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight
        var, mean = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - mean) / torch.sqrt(var + 1e-5)
        return F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)


def _conv3x3(cin: int, cout: int, stride: int = 1, bias: bool = False) -> StdConv2d:
    return StdConv2d(cin, cout, kernel_size=3, stride=stride, padding=1, bias=bias)


def _conv1x1(cin: int, cout: int, stride: int = 1, bias: bool = False) -> StdConv2d:
    return StdConv2d(cin, cout, kernel_size=1, stride=stride, padding=0, bias=bias)


class PreActBottleneck(nn.Module):
    """Pre-activation ResNetV2 bottleneck block."""

    def __init__(self, cin: int, cout: int | None = None, cmid: int | None = None, stride: int = 1):
        super().__init__()
        cout = cout or cin
        cmid = cmid or cout // 4

        self.gn1 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv1 = _conv1x1(cin, cmid, bias=False)
        self.gn2 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv2 = _conv3x3(cmid, cmid, stride=stride, bias=False)
        self.gn3 = nn.GroupNorm(32, cout, eps=1e-6)
        self.conv3 = _conv1x1(cmid, cout, bias=False)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or cin != cout:
            self.downsample = _conv1x1(cin, cout, stride=stride, bias=False)
            self.gn_proj = nn.GroupNorm(cout, cout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        if hasattr(self, "downsample"):
            residual = self.downsample(x)
            residual = self.gn_proj(residual)

        y = self.relu(self.gn1(self.conv1(x)))
        y = self.relu(self.gn2(self.conv2(y)))
        y = self.gn3(self.conv3(y))

        return self.relu(residual + y)

    def load_from(self, weights: Any, n_block: str, n_unit: str) -> None:
        """Load block weights from TransUNet npz checkpoint."""
        with torch.no_grad():
            _copy_param(
                self.conv1.weight,
                _np_to_torch(weights[f"{n_block}/{n_unit}/conv1/kernel"], conv=True),
            )
            _copy_param(
                self.conv2.weight,
                _np_to_torch(weights[f"{n_block}/{n_unit}/conv2/kernel"], conv=True),
            )
            _copy_param(
                self.conv3.weight,
                _np_to_torch(weights[f"{n_block}/{n_unit}/conv3/kernel"], conv=True),
            )

            _copy_param(
                self.gn1.weight,
                _np_to_torch(weights[f"{n_block}/{n_unit}/gn1/scale"]).view(-1),
            )
            _copy_param(
                self.gn1.bias,
                _np_to_torch(weights[f"{n_block}/{n_unit}/gn1/bias"]).view(-1),
            )

            _copy_param(
                self.gn2.weight,
                _np_to_torch(weights[f"{n_block}/{n_unit}/gn2/scale"]).view(-1),
            )
            _copy_param(
                self.gn2.bias,
                _np_to_torch(weights[f"{n_block}/{n_unit}/gn2/bias"]).view(-1),
            )

            _copy_param(
                self.gn3.weight,
                _np_to_torch(weights[f"{n_block}/{n_unit}/gn3/scale"]).view(-1),
            )
            _copy_param(
                self.gn3.bias,
                _np_to_torch(weights[f"{n_block}/{n_unit}/gn3/bias"]).view(-1),
            )

            if hasattr(self, "downsample"):
                _copy_param(
                    self.downsample.weight,
                    _np_to_torch(weights[f"{n_block}/{n_unit}/conv_proj/kernel"], conv=True),
                )
                _copy_param(
                    self.gn_proj.weight,
                    _np_to_torch(weights[f"{n_block}/{n_unit}/gn_proj/scale"]).view(-1),
                )
                _copy_param(
                    self.gn_proj.bias,
                    _np_to_torch(weights[f"{n_block}/{n_unit}/gn_proj/bias"]).view(-1),
                )


class ResNetV2(nn.Module):
    """ResNetV2 trunk used by TransUNet hybrid embeddings."""

    def __init__(self, block_units: tuple[int, int, int], width_factor: float):
        super().__init__()
        width = int(64 * width_factor)
        self.width = width

        self.root = nn.Sequential(
            OrderedDict(
                [
                    ("conv", StdConv2d(3, width, kernel_size=7, stride=2, bias=False, padding=3)),
                    ("gn", nn.GroupNorm(32, width, eps=1e-6)),
                    ("relu", nn.ReLU(inplace=True)),
                ]
            )
        )

        self.body = nn.Sequential(
            OrderedDict(
                [
                    (
                        "block1",
                        nn.Sequential(
                            OrderedDict(
                                [("unit1", PreActBottleneck(cin=width, cout=width * 4, cmid=width))]
                                + [
                                    (
                                        f"unit{i}",
                                        PreActBottleneck(
                                            cin=width * 4,
                                            cout=width * 4,
                                            cmid=width,
                                        ),
                                    )
                                    for i in range(2, block_units[0] + 1)
                                ]
                            )
                        ),
                    ),
                    (
                        "block2",
                        nn.Sequential(
                            OrderedDict(
                                [
                                    (
                                        "unit1",
                                        PreActBottleneck(
                                            cin=width * 4,
                                            cout=width * 8,
                                            cmid=width * 2,
                                            stride=2,
                                        ),
                                    )
                                ]
                                + [
                                    (
                                        f"unit{i}",
                                        PreActBottleneck(
                                            cin=width * 8,
                                            cout=width * 8,
                                            cmid=width * 2,
                                        ),
                                    )
                                    for i in range(2, block_units[1] + 1)
                                ]
                            )
                        ),
                    ),
                    (
                        "block3",
                        nn.Sequential(
                            OrderedDict(
                                [
                                    (
                                        "unit1",
                                        PreActBottleneck(
                                            cin=width * 8,
                                            cout=width * 16,
                                            cmid=width * 4,
                                            stride=2,
                                        ),
                                    )
                                ]
                                + [
                                    (
                                        f"unit{i}",
                                        PreActBottleneck(
                                            cin=width * 16,
                                            cout=width * 16,
                                            cmid=width * 4,
                                        ),
                                    )
                                    for i in range(2, block_units[2] + 1)
                                ]
                            )
                        ),
                    ),
                ]
            )
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        features: list[torch.Tensor] = []
        bsz, _, in_size, _ = x.size()

        x = self.root(x)
        features.append(x)
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)(x)

        for i in range(len(self.body) - 1):
            x = self.body[i](x)
            right_size = int(in_size / 4 / (i + 1))
            if x.size(2) != right_size:
                pad = right_size - x.size(2)
                if pad >= 3 or pad <= 0:
                    raise ValueError(
                        f"Unexpected feature size {x.shape}; expected spatial {right_size}"
                    )
                feat = torch.zeros(
                    (bsz, x.size(1), right_size, right_size),
                    device=x.device,
                    dtype=x.dtype,
                )
                feat[:, :, 0 : x.size(2), 0 : x.size(3)] = x
            else:
                feat = x
            features.append(feat)

        x = self.body[-1](x)
        return x, features[::-1]


def _swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


_ACT2FN = {
    "gelu": F.gelu,
    "relu": F.relu,
    "swish": _swish,
}


class Attention(nn.Module):
    """Multi-head self-attention used in ViT encoder blocks."""

    def __init__(self, config: TransUNetConfig, vis: bool):
        super().__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer.num_heads
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.out = nn.Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = nn.Dropout(config.transformer.attention_dropout_rate)
        self.proj_dropout = nn.Dropout(config.transformer.attention_dropout_rate)

        self.softmax = nn.Softmax(dim=-1)

    def _transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self._transpose_for_scores(mixed_query_layer)
        key_layer = self._transpose_for_scores(mixed_key_layer)
        value_layer = self._transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Mlp(nn.Module):
    """Transformer MLP block."""

    def __init__(self, config: TransUNetConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.transformer.mlp_dim)
        self.fc2 = nn.Linear(config.transformer.mlp_dim, config.hidden_size)
        self.act_fn = _ACT2FN["gelu"]
        self.dropout = nn.Dropout(config.transformer.dropout_rate)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Embeddings(nn.Module):
    """Patch + position embeddings for TransUNet transformer encoder."""

    def __init__(
        self,
        config: TransUNetConfig,
        img_size: int | tuple[int, int],
        in_channels: int = 3,
    ):
        super().__init__()
        self.hybrid = False
        self.config = config
        img_size = _pair(img_size)

        if config.patches_grid is not None:
            grid_size = config.patches_grid
            patch_size = (
                img_size[0] // 16 // grid_size[0],
                img_size[1] // 16 // grid_size[1],
            )
            if patch_size[0] <= 0 or patch_size[1] <= 0:
                raise ValueError(
                    "Invalid TransUNet patch size derived from image size and patches grid. "
                    f"img_size={img_size}, grid={grid_size}."
                )
            patch_size_real = (patch_size[0] * 16, patch_size[1] * 16)
            n_patches = (img_size[0] // patch_size_real[0]) * (img_size[1] // patch_size_real[1])
            self.hybrid = True
        else:
            raise ValueError("TransUNet v1 requires hybrid patches_grid configuration.")

        if self.hybrid:
            self.hybrid_model = ResNetV2(
                block_units=config.resnet.num_layers,
                width_factor=config.resnet.width_factor,
            )
            in_channels = self.hybrid_model.width * 16

        self.patch_embeddings = nn.Conv2d(
            in_channels=in_channels,
            out_channels=config.hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))
        self.dropout = nn.Dropout(config.transformer.dropout_rate)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        if self.hybrid:
            x, features = self.hybrid_model(x)
        else:
            features = None

        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings, features


_ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
_ATTENTION_K = "MultiHeadDotProductAttention_1/key"
_ATTENTION_V = "MultiHeadDotProductAttention_1/value"
_ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
_FC_0 = "MlpBlock_3/Dense_0"
_FC_1 = "MlpBlock_3/Dense_1"
_ATTENTION_NORM = "LayerNorm_0"
_MLP_NORM = "LayerNorm_2"


class Block(nn.Module):
    """Transformer encoder block with pre-norm attention + MLP."""

    def __init__(self, config: TransUNetConfig, vis: bool):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

    def load_from(self, weights: Any, n_block: int) -> None:
        """Load transformer block weights from npz checkpoint."""
        root = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = _np_to_torch(weights[f"{root}/{_ATTENTION_Q}/kernel"]).view(
                self.hidden_size, self.hidden_size
            ).t()
            key_weight = _np_to_torch(weights[f"{root}/{_ATTENTION_K}/kernel"]).view(
                self.hidden_size, self.hidden_size
            ).t()
            value_weight = _np_to_torch(weights[f"{root}/{_ATTENTION_V}/kernel"]).view(
                self.hidden_size, self.hidden_size
            ).t()
            out_weight = _np_to_torch(weights[f"{root}/{_ATTENTION_OUT}/kernel"]).view(
                self.hidden_size, self.hidden_size
            ).t()

            _copy_param(self.attn.query.weight, query_weight)
            _copy_param(self.attn.key.weight, key_weight)
            _copy_param(self.attn.value.weight, value_weight)
            _copy_param(self.attn.out.weight, out_weight)

            _copy_param(
                self.attn.query.bias,
                _np_to_torch(weights[f"{root}/{_ATTENTION_Q}/bias"]).view(-1),
            )
            _copy_param(
                self.attn.key.bias,
                _np_to_torch(weights[f"{root}/{_ATTENTION_K}/bias"]).view(-1),
            )
            _copy_param(
                self.attn.value.bias,
                _np_to_torch(weights[f"{root}/{_ATTENTION_V}/bias"]).view(-1),
            )
            _copy_param(
                self.attn.out.bias,
                _np_to_torch(weights[f"{root}/{_ATTENTION_OUT}/bias"]).view(-1),
            )

            _copy_param(self.ffn.fc1.weight, _np_to_torch(weights[f"{root}/{_FC_0}/kernel"]).t())
            _copy_param(self.ffn.fc2.weight, _np_to_torch(weights[f"{root}/{_FC_1}/kernel"]).t())
            _copy_param(self.ffn.fc1.bias, _np_to_torch(weights[f"{root}/{_FC_0}/bias"]).view(-1))
            _copy_param(self.ffn.fc2.bias, _np_to_torch(weights[f"{root}/{_FC_1}/bias"]).view(-1))

            _copy_param(
                self.attention_norm.weight,
                _np_to_torch(weights[f"{root}/{_ATTENTION_NORM}/scale"]),
            )
            _copy_param(
                self.attention_norm.bias,
                _np_to_torch(weights[f"{root}/{_ATTENTION_NORM}/bias"]),
            )
            _copy_param(
                self.ffn_norm.weight,
                _np_to_torch(weights[f"{root}/{_MLP_NORM}/scale"]),
            )
            _copy_param(
                self.ffn_norm.bias,
                _np_to_torch(weights[f"{root}/{_MLP_NORM}/bias"]),
            )


class Encoder(nn.Module):
    """Stacked transformer blocks."""

    def __init__(self, config: TransUNetConfig, vis: bool):
        super().__init__()
        self.vis = vis
        self.layers = nn.ModuleList(
            [Block(config, vis) for _ in range(config.transformer.num_layers)]
        )
        self.encoder_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        attn_weights: list[torch.Tensor] = []
        for layer_block in self.layers:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis and weights is not None:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class Transformer(nn.Module):
    """Hybrid embedding + transformer encoder."""

    def __init__(self, config: TransUNetConfig, img_size: int | tuple[int, int], vis: bool):
        super().__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config, vis)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor] | None]:
        embedding_output, features = self.embeddings(x)
        encoded, attn_weights = self.encoder(embedding_output)
        return encoded, attn_weights, features


class Conv2dReLU(nn.Sequential):
    """Conv-BN-ReLU helper used by decoder."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int = 0):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            bias=False,
        )
        bn = nn.BatchNorm2d(out_channels)
        relu = nn.ReLU(inplace=True)
        super().__init__(conv, bn, relu)


class DecoderBlock(nn.Module):
    """Upsampling decoder block with optional skip connection."""

    def __init__(self, in_channels: int, out_channels: int, skip_channels: int = 0):
        super().__init__()
        self.conv1 = Conv2dReLU(in_channels + skip_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = Conv2dReLU(out_channels, out_channels, kernel_size=3, padding=1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SegmentationHead(nn.Sequential):
    """Final segmentation projection head."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        upsampling: int = 1,
    ):
        conv2d = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        upsample = (
            nn.UpsamplingBilinear2d(scale_factor=upsampling)
            if upsampling > 1
            else nn.Identity()
        )
        super().__init__(conv2d, upsample)


class DecoderCup(nn.Module):
    """U-Net style decoder used by TransUNet."""

    def __init__(self, config: TransUNetConfig):
        super().__init__()
        self.config = config
        head_channels = 512
        self.conv_more = Conv2dReLU(config.hidden_size, head_channels, kernel_size=3, padding=1)

        decoder_channels = config.decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = list(decoder_channels)

        if config.n_skip != 0:
            skip_channels = list(config.skip_channels)
            for i in range(4 - config.n_skip):
                skip_channels[3 - i] = 0
        else:
            skip_channels = [0, 0, 0, 0]

        self.blocks = nn.ModuleList(
            [
                DecoderBlock(in_ch, out_ch, sk_ch)
                for in_ch, out_ch, sk_ch in zip(
                    in_channels,
                    out_channels,
                    skip_channels,
                    strict=False,
                )
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        features: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        bsz, n_patch, hidden = hidden_states.size()
        h = int(np.sqrt(n_patch))
        w = int(np.sqrt(n_patch))
        if h * w != n_patch:
            raise ValueError(
                "TransUNet expects square token grids. "
                f"Received n_patch={n_patch}, cannot reshape to square."
            )

        x = hidden_states.permute(0, 2, 1).contiguous().view(bsz, hidden, h, w)
        x = self.conv_more(x)

        for i, decoder_block in enumerate(self.blocks):
            skip = features[i] if (features is not None and i < self.config.n_skip) else None
            x = decoder_block(x, skip=skip)
        return x


class TransUNet(nn.Module):
    """TransUNet segmentation network for binary lesion segmentation.

    Args:
        config: Model architecture configuration.
        img_size: Input image size (H, W) or single integer for square size.
        in_channels: Number of input channels.
        vis: Whether to keep attention maps.
    """

    def __init__(
        self,
        config: TransUNetConfig,
        img_size: int | tuple[int, int] = 256,
        in_channels: int = 3,
        vis: bool = False,
    ):
        super().__init__()
        self.config = config
        self.transformer = Transformer(config, img_size, vis)
        self.decoder = DecoderCup(config)
        self.segmentation_head = SegmentationHead(
            in_channels=config.decoder_channels[-1],
            out_channels=config.n_classes,
            kernel_size=3,
        )
        self.img_size = _pair(img_size)
        self.in_channels = in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        x, _, features = self.transformer(x)
        x = self.decoder(x, features)
        logits = self.segmentation_head(x)
        return logits

    def _resize_positional_embeddings(
        self,
        posemb: torch.Tensor,
        ntok_new: int,
    ) -> torch.Tensor:
        """Resize position embeddings with bilinear interpolation in torch."""
        if posemb.ndim != 3:
            raise ValueError(
                f"Expected posemb with shape (1, N, C), got {tuple(posemb.shape)}"
            )

        if posemb.size(1) == ntok_new + 1:
            posemb = posemb[:, 1:]
        if posemb.size(1) == ntok_new:
            return posemb

        gs_old = int(math.sqrt(posemb.size(1)))
        gs_new = int(math.sqrt(ntok_new))
        if gs_old * gs_old != posemb.size(1) or gs_new * gs_new != ntok_new:
            raise ValueError(
                "Cannot resize positional embeddings for non-square token grids: "
                f"old_tokens={posemb.size(1)}, new_tokens={ntok_new}."
            )

        posemb_grid = posemb.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
        posemb_grid = F.interpolate(
            posemb_grid,
            size=(gs_new, gs_new),
            mode="bilinear",
            align_corners=False,
        )
        posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new * gs_new, -1)
        return posemb_grid

    def load_pretrained_from_npz(self, checkpoint_path: str | Path) -> None:
        """Load official ViT npz weights for R50-ViT-B_16.

        Args:
            checkpoint_path: Path to ``R50+ViT-B_16.npz``.

        Raises:
            ValueError: If required keys are missing or shape mapping is invalid.
        """
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            raise ValueError(f"TransUNet pretrained checkpoint not found: {ckpt_path}")

        try:
            weights = np.load(str(ckpt_path), allow_pickle=False)
        except Exception as exc:
            raise ValueError(f"Cannot read TransUNet pretrained checkpoint: {ckpt_path}") from exc

        try:
            with torch.no_grad():
                emb = self.transformer.embeddings

                _copy_param(
                    emb.patch_embeddings.weight,
                    _np_to_torch(weights["embedding/kernel"], conv=True),
                )
                _copy_param(emb.patch_embeddings.bias, _np_to_torch(weights["embedding/bias"]))

                _copy_param(
                    self.transformer.encoder.encoder_norm.weight,
                    _np_to_torch(weights["Transformer/encoder_norm/scale"]),
                )
                _copy_param(
                    self.transformer.encoder.encoder_norm.bias,
                    _np_to_torch(weights["Transformer/encoder_norm/bias"]),
                )

                posemb = _np_to_torch(weights["Transformer/posembed_input/pos_embedding"])
                posemb_new = emb.position_embeddings
                if posemb.size() != posemb_new.size():
                    posemb = self._resize_positional_embeddings(posemb, posemb_new.size(1))
                _copy_param(emb.position_embeddings, posemb)

                for block_idx, block in enumerate(self.transformer.encoder.layers):
                    block.load_from(weights, n_block=block_idx)

                if emb.hybrid:
                    _copy_param(
                        emb.hybrid_model.root.conv.weight,
                        _np_to_torch(weights["conv_root/kernel"], conv=True),
                    )
                    _copy_param(
                        emb.hybrid_model.root.gn.weight,
                        _np_to_torch(weights["gn_root/scale"]).view(-1),
                    )
                    _copy_param(
                        emb.hybrid_model.root.gn.bias,
                        _np_to_torch(weights["gn_root/bias"]).view(-1),
                    )

                    for bname, block in emb.hybrid_model.body.named_children():
                        for uname, unit in block.named_children():
                            unit.load_from(weights, n_block=bname, n_unit=uname)
        except KeyError as exc:
            raise ValueError(
                "TransUNet checkpoint missing required key. "
                f"Expected official R50+ViT-B_16.npz format, missing: {exc}"
            ) from exc
