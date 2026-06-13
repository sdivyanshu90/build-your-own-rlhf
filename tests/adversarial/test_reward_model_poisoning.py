"""Adversarial test: detectability of reward-model preference poisoning."""

from __future__ import annotations

import copy

import torch
from transformers import GPT2Config, PreTrainedTokenizerBase

from rlhf.config.schema import RewardModelConfig
from rlhf.data.preprocessing import encode_pair, pad_sequences
from rlhf.data.schemas import Preference
from rlhf.models.reward_model import RewardModelEnsemble
from rlhf.training.reward_model.ensemble import EnsembleRewardModelTrainer
from rlhf.training.reward_model.trainer import RewardModelTrainer


def _poison(preferences: list[Preference], rate: float) -> list[Preference]:
    """Return a copy of ``preferences`` with the first ``rate`` fraction swapped."""
    poisoned = copy.deepcopy(preferences)
    n_poison = int(len(poisoned) * rate)
    for pref in poisoned[:n_poison]:
        pref.chosen, pref.rejected = pref.rejected, pref.chosen
    return poisoned


def _final_loss(history: list[dict[str, float]], epochs: int) -> float:
    steps_per_epoch = max(len(history) // epochs, 1)
    last = history[-steps_per_epoch:]
    return sum(h["loss"] for h in last) / len(last)


def test_poisoning_raises_training_loss(
    lm_config: GPT2Config,
    tokenizer: PreTrainedTokenizerBase,
    dummy_preferences: list[Preference],
) -> None:
    from rlhf.models.reward_model import RewardModel

    subset = dummy_preferences[:60]
    config = RewardModelConfig(
        model_name_or_path="gpt2",
        freeze_backbone=False,
        learning_rate=1e-3,
        batch_size=12,
        epochs=3,
    )

    torch.manual_seed(0)
    clean_rm = RewardModel.from_config(lm_config, freeze_backbone=False)
    clean_hist = RewardModelTrainer(clean_rm, tokenizer, config).train(subset)

    torch.manual_seed(0)
    poison_rm = RewardModel.from_config(lm_config, freeze_backbone=False)
    poison_hist = RewardModelTrainer(poison_rm, tokenizer, config).train(_poison(subset, 0.2))

    # 20% contradictory labels raise the achievable Bradley-Terry loss floor.
    assert _final_loss(poison_hist, 3) > _final_loss(clean_hist, 3)


def test_ensemble_disagreement_elevated_under_poisoning(
    lm_config: GPT2Config,
    tokenizer: PreTrainedTokenizerBase,
    dummy_preferences: list[Preference],
) -> None:
    subset = dummy_preferences[:50]
    config = RewardModelConfig(
        model_name_or_path="gpt2",
        freeze_backbone=False,
        learning_rate=1e-3,
        batch_size=10,
        epochs=2,
        ensemble_size=2,
    )

    def _train_ensemble(prefs: list[Preference]) -> RewardModelEnsemble:
        ensemble = RewardModelEnsemble.from_config(
            lm_config, ensemble_size=2, freeze_backbone=False
        )
        EnsembleRewardModelTrainer(ensemble, tokenizer, config).train(prefs)
        return ensemble

    clean_ensemble = _train_ensemble(subset)
    poison_ensemble = _train_ensemble(_poison(subset, 0.25))

    # Evaluate ensemble disagreement (std) on a fixed set of response texts.
    eval_ids = [encode_pair(tokenizer, p.prompt, p.chosen, 64) for p in subset[:16]]
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
    ids, mask = pad_sequences(eval_ids, pad_id)

    with torch.no_grad():
        _, clean_std = clean_ensemble(ids, mask)
        _, poison_std = poison_ensemble(ids, mask)

    # Members trained on contradictory labels disagree more on the same inputs.
    assert float(poison_std.mean()) > float(clean_std.mean())
