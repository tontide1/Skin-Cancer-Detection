from __future__ import annotations

import pytest

from src.training.distributed import parse_torchrun_env, single_process_context


def test_single_process_context_defaults() -> None:
    ctx = single_process_context()
    assert ctx.enabled is False
    assert ctx.rank == 0
    assert ctx.world_size == 1
    assert ctx.local_rank == 0
    assert ctx.is_main_process is True


def test_parse_torchrun_env_success() -> None:
    ctx = parse_torchrun_env(
        {"RANK": "1", "WORLD_SIZE": "2", "LOCAL_RANK": "1"}
    )
    assert ctx.enabled is True
    assert ctx.rank == 1
    assert ctx.world_size == 2
    assert ctx.local_rank == 1
    assert ctx.is_main_process is False


def test_parse_torchrun_env_missing_var_raises() -> None:
    with pytest.raises(ValueError, match="Thiếu biến môi trường torchrun"):
        parse_torchrun_env({"RANK": "0", "WORLD_SIZE": "2"})


def test_parse_torchrun_env_invalid_rank_raises() -> None:
    with pytest.raises(ValueError, match="không hợp lệ"):
        parse_torchrun_env({"RANK": "2", "WORLD_SIZE": "2", "LOCAL_RANK": "0"})
