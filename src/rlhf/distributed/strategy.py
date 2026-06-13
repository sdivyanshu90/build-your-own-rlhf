"""
distributed.strategy — DDP / FSDP / single-device training strategy abstraction.

Overview
--------
:class:`DistributedStrategy` hides the differences between three parallelism
modes behind one ``prepare_model`` call:

* ``SINGLE`` — no wrapping (single GPU or CPU); the default.
* ``DDP``    — ``DistributedDataParallel`` for data-parallel replicas.
* ``FSDP``   — ``FullyShardedDataParallel`` for sharded large-model training.

Wrapping only happens when a process group is actually initialized, so requesting
DDP/FSDP on a single process degrades cleanly to single-device behaviour rather
than crashing — keeping the training code identical across environments.
"""

from __future__ import annotations

import logging
from enum import Enum

import torch
from torch import nn

from rlhf.distributed.utils import get_rank, is_distributed

logger = logging.getLogger(__name__)


class StrategyType(str, Enum):
    """Supported parallelism strategies."""

    SINGLE = "single"
    DDP = "ddp"
    FSDP = "fsdp"


class DistributedStrategy:
    """Wraps models according to the selected parallelism strategy."""

    def __init__(self, strategy: StrategyType = StrategyType.SINGLE) -> None:
        self.strategy = strategy

    @property
    def device(self) -> torch.device:
        """The device this process should place its model shard on."""
        if torch.cuda.is_available():
            return torch.device(f"cuda:{get_rank() % max(torch.cuda.device_count(), 1)}")
        return torch.device("cpu")

    def prepare_model(self, model: nn.Module) -> nn.Module:
        """
        Move ``model`` to this process's device and wrap it for the strategy.

        Returns the model unwrapped when no process group is active (single
        process), so the call is safe everywhere.
        """
        model = model.to(self.device)
        if self.strategy == StrategyType.SINGLE or not is_distributed():
            if self.strategy != StrategyType.SINGLE:
                logger.warning(
                    "Strategy %s requested but no process group is initialized; "
                    "running single-device.",
                    self.strategy.value,
                )
            return model
        if self.strategy == StrategyType.DDP:
            return self._wrap_ddp(model)
        return self._wrap_fsdp(model)

    def _wrap_ddp(self, model: nn.Module) -> nn.Module:
        from torch.nn.parallel import DistributedDataParallel

        device_ids = [self.device.index] if self.device.type == "cuda" else None
        return DistributedDataParallel(model, device_ids=device_ids)

    def _wrap_fsdp(self, model: nn.Module) -> nn.Module:  # pragma: no cover - needs >=2 GPUs
        from torch.distributed.fsdp import FullyShardedDataParallel

        return FullyShardedDataParallel(model)


__all__ = ["DistributedStrategy", "StrategyType"]
