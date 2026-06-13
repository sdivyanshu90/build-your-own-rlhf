"""Integration test: Bradley-Terry reward-model training."""

from __future__ import annotations

from transformers import PreTrainedTokenizerBase

from rlhf.config.schema import RewardModelConfig
from rlhf.data.schemas import Preference
from rlhf.models.reward_model import RewardModel
from rlhf.training.reward_model.trainer import RewardModelTrainer


def test_reward_model_training_separates_preferences(
    text_reward_model: RewardModel,
    tokenizer: PreTrainedTokenizerBase,
    dummy_preferences: list[Preference],
) -> None:
    config = RewardModelConfig(
        model_name_or_path="gpt2",
        freeze_backbone=False,
        learning_rate=1e-3,
        batch_size=16,
        epochs=3,
    )
    trainer = RewardModelTrainer(text_reward_model, tokenizer, config)
    history = trainer.train(dummy_preferences)

    # Bradley-Terry loss decreases from the first epoch to the last.
    steps_per_epoch = len(history) // config.epochs
    first_epoch = history[:steps_per_epoch]
    last_epoch = history[-steps_per_epoch:]
    avg_first = sum(h["loss"] for h in first_epoch) / len(first_epoch)
    avg_last = sum(h["loss"] for h in last_epoch) / len(last_epoch)
    assert avg_last < avg_first

    # The model ranks chosen above rejected on the large majority of pairs.
    eval_metrics = trainer.evaluate(dummy_preferences)
    assert eval_metrics["accuracy"] > 0.7

    # The running normalizer accumulated non-trivial reward statistics.
    assert float(text_reward_model.normalizer.count) > 0
    assert float(text_reward_model.normalizer.variance) > 0.0
