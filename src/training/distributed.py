from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class DistributedContext:
    """Runtime context for single-process or torchrun-based DDP."""

    enabled: bool = False
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0


def single_process_context() -> DistributedContext:
    """Return default context for non-distributed training."""
    return DistributedContext(enabled=False, rank=0, world_size=1, local_rank=0)


def parse_torchrun_env(env: Mapping[str, str] | None = None) -> DistributedContext:
    """
    Parse torchrun environment variables into a validated context.

    Required variables: RANK, WORLD_SIZE, LOCAL_RANK.
    """
    env_map = os.environ if env is None else env
    required = ("RANK", "WORLD_SIZE", "LOCAL_RANK")
    missing = [key for key in required if key not in env_map]
    if missing:
        raise ValueError(
            "Thiếu biến môi trường torchrun: "
            f"{', '.join(missing)}. Hãy launch bằng torchrun."
        )

    try:
        rank = int(env_map["RANK"])
        world_size = int(env_map["WORLD_SIZE"])
        local_rank = int(env_map["LOCAL_RANK"])
    except ValueError as exc:
        raise ValueError("RANK/WORLD_SIZE/LOCAL_RANK phải là số nguyên.") from exc

    if world_size <= 0:
        raise ValueError("WORLD_SIZE phải > 0.")
    if rank < 0 or rank >= world_size:
        raise ValueError(
            f"RANK={rank} không hợp lệ cho WORLD_SIZE={world_size}."
        )
    if local_rank < 0:
        raise ValueError("LOCAL_RANK phải >= 0.")

    return DistributedContext(
        enabled=(world_size > 1),
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
    )
