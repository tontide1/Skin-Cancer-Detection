from __future__ import annotations
import pytest
from scripts.benchmark_fps import parse_args

def test_parse_args_defaults() -> None:
    args = parse_args(["--config", "cfg.yaml", "--checkpoint", "ckpt.pth"])
    assert args.config == "cfg.yaml"
    assert args.checkpoint == "ckpt.pth"
    assert args.batch_sizes == [1, 100, 1000]
    assert args.device == "cpu"

def test_parse_args_custom_batch_sizes() -> None:
    args = parse_args(["--config", "cfg.yaml", "--checkpoint", "ckpt.pth", "--batch-sizes", "1", "8", "16"])
    assert args.batch_sizes == [1, 8, 16]

def test_parse_args_overrides() -> None:
    args = parse_args(["--config", "cfg.yaml", "--checkpoint", "ckpt.pth", "model.name=unet", "data.batch_size=32"])
    assert args.overrides == ["model.name=unet", "data.batch_size=32"]
