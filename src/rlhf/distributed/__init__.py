"""distributed — DDP / FSDP strategy abstraction and process-group helpers."""

from __future__ import annotations

from rlhf.distributed.strategy import DistributedStrategy, StrategyType
from rlhf.distributed.utils import (
    all_reduce_mean,
    barrier,
    cleanup_process_group,
    get_rank,
    get_world_size,
    is_distributed,
    is_main_process,
    setup_process_group,
)

__all__ = [
    "DistributedStrategy",
    "StrategyType",
    "all_reduce_mean",
    "barrier",
    "cleanup_process_group",
    "get_rank",
    "get_world_size",
    "is_distributed",
    "is_main_process",
    "setup_process_group",
]
