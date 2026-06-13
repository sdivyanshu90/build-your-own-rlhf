"""Unit tests for the LR scheduler factory."""

from __future__ import annotations

import pytest
import torch

from rlhf.exceptions import ConfigError
from rlhf.training.ppo.scheduler import build_lr_scheduler


def _optimizer() -> torch.optim.Optimizer:
    return torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)


def test_warmup_ramps_linearly() -> None:
    opt = _optimizer()
    sched = build_lr_scheduler(opt, "constant", warmup_steps=4, total_steps=10)
    lrs = [opt.param_groups[0]["lr"]]
    for _ in range(4):
        sched.step()
        lrs.append(opt.param_groups[0]["lr"])
    # LR rises from 0 toward the peak across the warmup window.
    assert lrs[0] == pytest.approx(0.0)
    assert lrs[-1] == pytest.approx(1.0)
    assert lrs == sorted(lrs)


def test_cosine_decays_to_floor() -> None:
    opt = _optimizer()
    sched = build_lr_scheduler(opt, "cosine", warmup_steps=0, total_steps=10, min_lr_ratio=0.1)
    for _ in range(10):
        sched.step()
    # After the full horizon the cosine schedule reaches the floor.
    assert opt.param_groups[0]["lr"] == pytest.approx(0.1, abs=1e-6)


def test_linear_decays() -> None:
    opt = _optimizer()
    sched = build_lr_scheduler(opt, "linear", warmup_steps=0, total_steps=10, min_lr_ratio=0.0)
    first = opt.param_groups[0]["lr"]
    for _ in range(5):
        sched.step()
    assert opt.param_groups[0]["lr"] < first


def test_invalid_scheduler_type_raises() -> None:
    with pytest.raises(ConfigError):
        build_lr_scheduler(_optimizer(), "exponential", 0, 10)  # type: ignore[arg-type]


def test_invalid_min_lr_ratio_raises() -> None:
    with pytest.raises(ConfigError):
        build_lr_scheduler(_optimizer(), "cosine", 0, 10, min_lr_ratio=2.0)


def test_negative_warmup_raises() -> None:
    with pytest.raises(ConfigError):
        build_lr_scheduler(_optimizer(), "cosine", -1, 10)
