"""training.ppo — the from-scratch Proximal Policy Optimization implementation."""

from __future__ import annotations

from rlhf.training.ppo.algorithm import PPOLossOutput, compute_ppo_loss
from rlhf.training.ppo.kl_controller import (
    AbstractKLController,
    AdaptiveKLController,
    FixedKLController,
    make_kl_controller,
)
from rlhf.training.ppo.rollout import (
    RolloutBuffer,
    RolloutMiniBatch,
    compute_gae,
    compute_rlhf_rewards,
    whiten,
)
from rlhf.training.ppo.scheduler import build_lr_scheduler
from rlhf.training.ppo.trainer import PPOTrainer

__all__ = [
    "AbstractKLController",
    "AdaptiveKLController",
    "FixedKLController",
    "PPOLossOutput",
    "PPOTrainer",
    "RolloutBuffer",
    "RolloutMiniBatch",
    "build_lr_scheduler",
    "compute_gae",
    "compute_ppo_loss",
    "compute_rlhf_rewards",
    "make_kl_controller",
    "whiten",
]
