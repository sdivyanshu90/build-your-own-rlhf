"""Integration test: a 5+5 split run reproduces a 10-step continuous run."""

from __future__ import annotations

import pytest
from transformers import GPT2Config, PreTrainedTokenizerBase

from rlhf.config.schema import PPOConfig
from rlhf.models.policy import PolicyModel
from rlhf.models.reference_model import ReferenceModel
from rlhf.models.reward_model import RewardModel
from rlhf.training.ppo.trainer import PPOTrainer
from rlhf.utils import set_seed

PROMPTS = [
    "The capital of France is",
    "Once upon a time",
    "In the beginning there was",
    "Two plus two equals",
]
_WEIGHT_SEED = 123
_RUN_SEED = 7


def _build_models(lm_config: GPT2Config) -> tuple[PolicyModel, ReferenceModel, RewardModel]:
    """Construct identically-initialized models (deterministic given the seed)."""
    set_seed(_WEIGHT_SEED)
    policy = PolicyModel.from_config(lm_config, value_head_dropout=0.0)
    reference = ReferenceModel.from_policy(policy)
    reward_model = RewardModel.from_config(lm_config, freeze_backbone=False)
    return policy, reference, reward_model


def _make_trainer(
    lm_config: GPT2Config, tokenizer: PreTrainedTokenizerBase, out: str
) -> PPOTrainer:
    policy, reference, reward_model = _build_models(lm_config)
    config = PPOConfig(
        rollout_batch_size=4,
        mini_batch_size=2,
        ppo_epochs=2,
        max_new_tokens=6,
        learning_rate=1e-3,
        warmup_steps=0,
        total_steps=10,
        save_every=1000,
    )
    return PPOTrainer(
        policy, reference, tokenizer, config, PROMPTS, reward_model=reward_model, output_dir=out
    )


def test_checkpoint_resume_matches_continuous(
    lm_config: GPT2Config,
    tokenizer: PreTrainedTokenizerBase,
    tmp_path,  # type: ignore[no-untyped-def]
) -> None:
    # --- Continuous run: 10 steps in one go. ---
    continuous = _make_trainer(lm_config, tokenizer, str(tmp_path / "cont"))
    set_seed(_RUN_SEED)
    cont_metrics = [continuous.train_step() for _ in range(10)]

    # --- Split run: 5 steps, checkpoint, fresh trainer restores, 5 more steps. ---
    first_half = _make_trainer(lm_config, tokenizer, str(tmp_path / "split"))
    set_seed(_RUN_SEED)
    for _ in range(5):
        first_half.train_step()
    ckpt_dir = tmp_path / "ckpt"
    first_half.save_checkpoint(ckpt_dir)

    resumed = _make_trainer(lm_config, tokenizer, str(tmp_path / "resumed"))
    resumed.load_checkpoint(ckpt_dir)
    # State restored from the checkpoint (step, KL coeff, optimizer moments).
    assert resumed.global_step == 5
    assert resumed.kl_controller.value == pytest.approx(first_half.kl_controller.value)
    assert resumed.scheduler.last_epoch == first_half.scheduler.last_epoch

    resume_metrics = [resumed.train_step() for _ in range(5)]

    # The 10th step's metrics must match between continuous and resumed runs.
    for key in ("train/total_loss", "train/policy_loss", "train/value_loss", "reward/mean"):
        assert resume_metrics[-1][key] == pytest.approx(cont_metrics[-1][key], abs=1e-4), key
