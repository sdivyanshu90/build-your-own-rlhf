"""
training.ppo.scheduler — learning-rate schedules with linear warmup.

Overview
--------
:func:`build_lr_scheduler` returns a ``torch.optim.lr_scheduler.LambdaLR`` that
linearly warms up the learning rate for ``warmup_steps`` and then decays it with
one of three schedules:

* ``constant`` — hold the peak LR after warmup.
* ``linear``   — linearly decay from peak to ``min_lr_ratio * peak``.
* ``cosine``   — cosine decay from peak to ``min_lr_ratio * peak``.

Keeping this as a pure factory (no global state) means the schedule is fully
reconstructible from config — important for deterministic checkpoint resume.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Literal

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from rlhf.exceptions import ConfigError

SchedulerType = Literal["cosine", "linear", "constant"]


def _make_lambda(
    scheduler_type: SchedulerType,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float,
) -> Callable[[int], float]:
    """Build the per-step multiplier function for ``LambdaLR``."""
    # The decay denominator is the number of post-warmup steps; guard against a
    # zero/negative span when total_steps <= warmup_steps.
    decay_steps = max(total_steps - warmup_steps, 1)

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            # Linear warmup from 0 -> 1 over the first warmup_steps.
            return float(step) / float(max(warmup_steps, 1))
        progress = float(step - warmup_steps) / float(decay_steps)
        progress = min(max(progress, 0.0), 1.0)
        if scheduler_type == "constant":
            return 1.0
        if scheduler_type == "linear":
            return min_lr_ratio + (1.0 - min_lr_ratio) * (1.0 - progress)
        # cosine
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return lr_lambda


def build_lr_scheduler(
    optimizer: Optimizer,
    scheduler_type: SchedulerType,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.0,
) -> LambdaLR:
    """
    Construct a warmup + decay LR scheduler.

    Args:
        optimizer: The optimizer whose LR to schedule.
        scheduler_type: ``"cosine"``, ``"linear"`` or ``"constant"``.
        warmup_steps: Number of linear-warmup steps.
        total_steps: Total training steps (for the decay horizon).
        min_lr_ratio: Floor on the LR as a fraction of the peak.

    Returns:
        A configured :class:`LambdaLR`.
    """
    if scheduler_type not in ("cosine", "linear", "constant"):
        raise ConfigError(f"Unknown scheduler_type '{scheduler_type}'.")
    if warmup_steps < 0:
        raise ConfigError("warmup_steps must be non-negative.")
    if not (0.0 <= min_lr_ratio <= 1.0):
        raise ConfigError("min_lr_ratio must lie in [0, 1].")
    lr_lambda = _make_lambda(scheduler_type, warmup_steps, total_steps, min_lr_ratio)
    return LambdaLR(optimizer, lr_lambda)


__all__ = ["SchedulerType", "build_lr_scheduler"]
