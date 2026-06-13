"""Integration test: FSDP wrapping (skipped unless 2+ GPUs are present)."""

from __future__ import annotations

import pytest
import torch

from rlhf.distributed.strategy import DistributedStrategy, StrategyType

pytestmark = pytest.mark.gpu

_MIN_GPUS = 2


@pytest.mark.skipif(
    torch.cuda.device_count() < _MIN_GPUS,
    reason="FSDP test requires at least 2 GPUs",
)
def test_fsdp_strategy_wraps_model(text_policy) -> None:  # type: ignore[no-untyped-def]
    # Exercised only on multi-GPU runners; verifies the FSDP strategy wraps a
    # model and exposes a usable optimizer/parameter set.
    strategy = DistributedStrategy(StrategyType.FSDP)
    wrapped = strategy.prepare_model(text_policy)
    assert wrapped is not None
    assert any(p.requires_grad for p in wrapped.parameters())
