"""
distributed.utils — process-group helpers for multi-GPU training.

Overview
--------
Thin wrappers over ``torch.distributed`` that degrade gracefully to single-process
behaviour when no process group is initialized — so the same training code runs
unchanged on a laptop CPU and on a multi-node GPU cluster.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist


def is_distributed() -> bool:
    """True when a ``torch.distributed`` process group is initialized."""
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    """Global rank of this process (0 when not distributed)."""
    return dist.get_rank() if is_distributed() else 0


def get_world_size() -> int:
    """Total number of processes (1 when not distributed)."""
    return dist.get_world_size() if is_distributed() else 1


def is_main_process() -> bool:
    """True on the rank-0 process (always True when not distributed)."""
    return get_rank() == 0


def barrier() -> None:
    """Synchronize all processes (no-op when not distributed)."""
    if is_distributed():
        dist.barrier()


def setup_process_group(backend: str = "nccl") -> None:
    """
    Initialize the default process group from standard env vars.

    Expects ``RANK``, ``WORLD_SIZE`` and ``LOCAL_RANK`` to be set (as torchrun
    does). No-op if a group is already initialized or the world size is 1.
    """
    if is_distributed():
        return
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return
    chosen = backend if (backend == "gloo" or torch.cuda.is_available()) else "gloo"
    dist.init_process_group(backend=chosen)
    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))


def cleanup_process_group() -> None:
    """Destroy the default process group if one is initialized."""
    if is_distributed():
        dist.destroy_process_group()


def all_reduce_mean(value: torch.Tensor) -> torch.Tensor:
    """Average a tensor across all processes (identity when not distributed)."""
    if not is_distributed():
        return value
    reduced = value.clone()
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    return reduced / get_world_size()


__all__ = [
    "all_reduce_mean",
    "barrier",
    "cleanup_process_group",
    "get_rank",
    "get_world_size",
    "is_distributed",
    "is_main_process",
    "setup_process_group",
]
