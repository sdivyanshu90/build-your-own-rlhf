"""Unit tests for the distributed strategy and process-group helpers (CPU paths)."""

from __future__ import annotations

import torch

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


def test_single_process_helpers() -> None:
    assert is_distributed() is False
    assert get_rank() == 0
    assert get_world_size() == 1
    assert is_main_process() is True
    barrier()  # no-op
    cleanup_process_group()  # no-op
    # setup is a no-op when WORLD_SIZE is unset / 1.
    setup_process_group(backend="gloo")


def test_all_reduce_mean_identity_when_single() -> None:
    x = torch.tensor([1.0, 2.0, 3.0])
    assert torch.equal(all_reduce_mean(x), x)


def test_strategy_single_returns_model_unwrapped() -> None:
    strategy = DistributedStrategy(StrategyType.SINGLE)
    model = torch.nn.Linear(4, 4)
    prepared = strategy.prepare_model(model)
    assert prepared is model
    # The device follows hardware availability (CUDA when present, else CPU).
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert strategy.device.type == expected


def test_strategy_ddp_falls_back_when_not_distributed() -> None:
    # Requesting DDP/FSDP without an initialized process group must degrade to
    # single-device rather than crash.
    for strategy_type in (StrategyType.DDP, StrategyType.FSDP):
        strategy = DistributedStrategy(strategy_type)
        model = torch.nn.Linear(2, 2)
        prepared = strategy.prepare_model(model)
        assert any(p.requires_grad for p in prepared.parameters())
