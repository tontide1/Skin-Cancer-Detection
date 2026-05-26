from __future__ import annotations

from types import SimpleNamespace

import torch

import scripts.benchmark_fps as benchmark_fps


def test_parse_args_defaults() -> None:
    args = benchmark_fps.parse_args(["--config", "cfg.yaml", "--checkpoint", "ckpt.pth"])
    assert args.config == "cfg.yaml"
    assert args.checkpoint == "ckpt.pth"
    assert args.batch_sizes == [1, 50, 100]
    assert args.device == "cpu"


def test_parse_args_custom_batch_sizes() -> None:
    args = benchmark_fps.parse_args(
        ["--config", "cfg.yaml", "--checkpoint", "ckpt.pth", "--batch-sizes", "1", "8", "16"]
    )
    assert args.batch_sizes == [1, 8, 16]


def test_parse_args_overrides() -> None:
    args = benchmark_fps.parse_args(
        ["--config", "cfg.yaml", "--checkpoint", "ckpt.pth", "model.name=unet", "data.batch_size=32"]
    )
    assert args.overrides == ["model.name=unet", "data.batch_size=32"]


def test_benchmark_batch_size_skips_oom(monkeypatch) -> None:
    class FakeTimer:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def blocked_autorange(self, min_run_time: float) -> SimpleNamespace:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")

    empty_cache_calls = {"count": 0}

    monkeypatch.setattr(benchmark_fps.benchmark, "Timer", FakeTimer)
    monkeypatch.setattr(benchmark_fps.gc, "collect", lambda: None)
    monkeypatch.setattr(
        benchmark_fps.torch.cuda,
        "empty_cache",
        lambda: empty_cache_calls.__setitem__("count", empty_cache_calls["count"] + 1),
    )

    result = benchmark_fps._benchmark_batch_size(
        model=SimpleNamespace(),
        device=torch.device("cpu"),
        height=8,
        width=8,
        batch_size=4,
    )

    assert result is None
    assert empty_cache_calls["count"] == 0
