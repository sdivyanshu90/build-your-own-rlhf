"""training.reward_model — Bradley-Terry reward-model and ensemble trainers."""

from __future__ import annotations

from rlhf.training.reward_model.ensemble import EnsembleRewardModelTrainer
from rlhf.training.reward_model.trainer import RewardModelTrainer

__all__ = ["EnsembleRewardModelTrainer", "RewardModelTrainer"]
