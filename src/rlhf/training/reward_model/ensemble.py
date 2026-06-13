"""
training.reward_model.ensemble — train an ensemble of reward models.

Overview
--------
:class:`EnsembleRewardModelTrainer` trains each member of a
:class:`~rlhf.models.reward_model.RewardModelEnsemble` independently (different
seeds, same data) so their disagreement at PPO time is a meaningful uncertainty
signal and a reward-hacking canary. Members are trained sequentially to keep peak
memory bounded; the per-member trainers are ordinary
:class:`~rlhf.training.reward_model.trainer.RewardModelTrainer` instances.
"""

from __future__ import annotations

import logging

import torch
from transformers import PreTrainedTokenizerBase

from rlhf.config.schema import RewardModelConfig
from rlhf.data.schemas import Preference
from rlhf.models.reward_model import RewardModelEnsemble
from rlhf.monitoring.logger import RLHFLogger
from rlhf.training.reward_model.trainer import RewardModelTrainer
from rlhf.utils import set_seed

logger = logging.getLogger(__name__)


class EnsembleRewardModelTrainer:
    """Sequentially trains every member of a reward-model ensemble."""

    def __init__(
        self,
        ensemble: RewardModelEnsemble,
        tokenizer: PreTrainedTokenizerBase,
        config: RewardModelConfig,
        device: torch.device | None = None,
        logger_backend: RLHFLogger | None = None,
    ) -> None:
        self.ensemble = ensemble
        self.tokenizer = tokenizer
        self.config = config
        self.device = device or torch.device("cpu")
        self.logger = logger_backend

    def train(self, preferences: list[Preference]) -> list[list[dict[str, float]]]:
        """
        Train all members; returns one training history per member.

        Each member is seeded distinctly so they explore different optima.
        """
        histories: list[list[dict[str, float]]] = []
        for i, member in enumerate(self.ensemble.models):
            # Distinct seed per member -> diverse heads -> meaningful ensemble std.
            set_seed(self.config.seed + i)
            trainer = RewardModelTrainer(
                member, self.tokenizer, self.config, self.device, self.logger
            )
            logger.info("Training ensemble member %d/%d", i + 1, self.ensemble.size)
            histories.append(trainer.train(preferences))
        return histories


__all__ = ["EnsembleRewardModelTrainer"]
