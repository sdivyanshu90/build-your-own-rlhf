"""
training.reward_model.trainer — Bradley-Terry reward-model training.

Overview
--------
:class:`RewardModelTrainer` trains a :class:`~rlhf.models.reward_model.RewardModel`
on preference pairs by maximizing the Bradley-Terry likelihood that the chosen
response out-scores the rejected one. After training it folds the observed reward
distribution into the model's running normalizer so PPO sees standardized rewards.

Mathematical Background
-----------------------
    L_BT = -E[ log sigmoid( r(x, y_w) - r(x, y_l) ) ]

The accuracy metric is the fraction of pairs with ``r(y_w) > r(y_l)``.

Usage Example
-------------
>>> # trainer = RewardModelTrainer(reward_model, tokenizer, RewardModelConfig(...))
>>> # history = trainer.train(preferences)
"""

from __future__ import annotations

import logging

import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

from rlhf.config.schema import RewardModelConfig
from rlhf.data.collators import PreferenceCollator
from rlhf.data.preference_dataset import PreferenceDataset
from rlhf.data.schemas import Preference
from rlhf.models.reward_model import RewardModel, bradley_terry_loss
from rlhf.monitoring.logger import RLHFLogger
from rlhf.utils import set_seed

logger = logging.getLogger(__name__)


class RewardModelTrainer:
    """Trainer for a single Bradley-Terry reward model."""

    def __init__(
        self,
        reward_model: RewardModel,
        tokenizer: PreTrainedTokenizerBase,
        config: RewardModelConfig,
        device: torch.device | None = None,
        logger_backend: RLHFLogger | None = None,
    ) -> None:
        self.model = reward_model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device or torch.device("cpu")
        self.model.to(self.device)
        self.logger = logger_backend
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = AdamW(trainable, lr=config.learning_rate)
        self.collator = PreferenceCollator(tokenizer)
        self.global_step = 0

    def _score(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """Score a (right-padded) batch, returning raw rewards (no normalization)."""
        # Training always uses raw rewards; normalization is a PPO-time concern.
        was_normalizing = self.model.normalize_rewards
        self.model.normalize_rewards = False
        try:
            rewards: Tensor = self.model(input_ids.to(self.device), attention_mask.to(self.device))
        finally:
            self.model.normalize_rewards = was_normalizing
        return rewards

    def train_step(self, batch: dict[str, Tensor]) -> dict[str, float]:
        """Run one Bradley-Terry optimization step; returns loss + accuracy."""
        self.model.train()
        chosen = self._score(batch["chosen_input_ids"], batch["chosen_attention_mask"])
        rejected = self._score(batch["rejected_input_ids"], batch["rejected_attention_mask"])
        loss = bradley_terry_loss(chosen, rejected)
        self.optimizer.zero_grad()
        loss.backward()  # type: ignore[no-untyped-call]  # torch stub gap
        nn.utils.clip_grad_norm_(
            (p for p in self.model.parameters() if p.requires_grad), self.config.max_grad_norm
        )
        self.optimizer.step()
        self.global_step += 1
        accuracy = float((chosen > rejected).float().mean())
        # Track the reward scale so PPO can later standardize rewards.
        self.model.update_normalizer(torch.cat([chosen.detach(), rejected.detach()]))
        return {"loss": float(loss.detach()), "accuracy": accuracy}

    def train(self, preferences: list[Preference]) -> list[dict[str, float]]:
        """
        Train for ``config.epochs`` epochs over ``preferences``.

        Returns:
            A list of per-step ``{"loss": ..., "accuracy": ...}`` dicts.
        """
        set_seed(self.config.seed)
        dataset = PreferenceDataset(preferences, self.tokenizer)
        loader: DataLoader[dict[str, list[int]]] = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=self.collator,
        )
        history: list[dict[str, float]] = []
        for epoch in range(self.config.epochs):
            for batch in loader:
                metrics = self.train_step(batch)
                history.append(metrics)
                if self.logger is not None:
                    self.logger.log_scalars(self.global_step, metrics, prefix="reward/")
            logger.info(
                "RM epoch %d: loss=%.4f acc=%.3f",
                epoch,
                history[-1]["loss"],
                history[-1]["accuracy"],
            )
        return history

    @torch.no_grad()
    def evaluate(self, preferences: list[Preference]) -> dict[str, float]:
        """Compute mean loss and accuracy over ``preferences`` without updating."""
        self.model.eval()
        dataset = PreferenceDataset(preferences, self.tokenizer)
        loader: DataLoader[dict[str, list[int]]] = DataLoader(
            dataset, batch_size=self.config.batch_size, collate_fn=self.collator
        )
        losses, accs, n = 0.0, 0.0, 0
        for batch in loader:
            chosen = self._score(batch["chosen_input_ids"], batch["chosen_attention_mask"])
            rejected = self._score(batch["rejected_input_ids"], batch["rejected_attention_mask"])
            bsz = chosen.shape[0]
            losses += float(bradley_terry_loss(chosen, rejected)) * bsz
            accs += float((chosen > rejected).float().sum())
            n += bsz
        return {"loss": losses / max(n, 1), "accuracy": accs / max(n, 1)}


__all__ = ["RewardModelTrainer"]
