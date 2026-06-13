"""Integration test: a short PPO training loop on a tiny policy."""

from __future__ import annotations

import math

import torch
from transformers import PreTrainedTokenizerBase

from rlhf.config.schema import PPOConfig
from rlhf.models.policy import PolicyModel
from rlhf.models.reference_model import ReferenceModel
from rlhf.models.reward_model import RewardModel
from rlhf.monitoring.logger import RLHFLogger
from rlhf.training.ppo.trainer import PPOTrainer

_EXPECTED_METRICS = [
    "train/policy_loss",
    "train/value_loss",
    "train/entropy",
    "train/total_loss",
    "train/kl_divergence",
    "train/approx_kl",
    "train/clip_fraction",
    "train/explained_variance",
    "train/gradient_norm",
    "train/learning_rate",
    "train/kl_coeff",
    "reward/mean",
    "reward/std",
    "reward/hacking_score",
    "response/length_mean",
]

PROMPTS = [
    "The capital of France is",
    "Once upon a time",
    "In the beginning there was",
    "Two plus two equals",
    "The quick brown fox",
    "To be or not to be",
]


def _build_trainer(
    policy: PolicyModel,
    reference: ReferenceModel,
    reward_model: RewardModel,
    tokenizer: PreTrainedTokenizerBase,
    tmp_path: str,
) -> PPOTrainer:
    config = PPOConfig(
        rollout_batch_size=6,
        mini_batch_size=3,
        ppo_epochs=2,
        max_new_tokens=8,
        learning_rate=1e-3,
        warmup_steps=0,
        kl_abort_threshold=30.0,
        total_steps=10,
        save_every=1000,
    )
    return PPOTrainer(
        policy,
        reference,
        tokenizer,
        config,
        PROMPTS,
        reward_model=reward_model,
        logger_backend=RLHFLogger(run_name="test"),
        output_dir=tmp_path,
    )


def test_ppo_training_loop_smoke(
    text_policy: PolicyModel,
    text_reference: ReferenceModel,
    text_reward_model: RewardModel,
    tokenizer: PreTrainedTokenizerBase,
    tmp_path,  # type: ignore[no-untyped-def]
) -> None:
    trainer = _build_trainer(
        text_policy, text_reference, text_reward_model, tokenizer, str(tmp_path)
    )
    params_before = [p.detach().clone() for p in trainer.policy.parameters() if p.requires_grad]

    history = trainer.train(num_steps=10)
    assert len(history) == 10

    for metrics in history:
        # Every expected metric is present and finite at every step.
        for key in _EXPECTED_METRICS:
            assert key in metrics, key
            assert math.isfinite(metrics[key]), (key, metrics[key])
        # KL must remain under the abort threshold.
        assert metrics["train/kl_divergence"] < trainer.config.kl_abort_threshold

    # Learning signal: the value head's output layer is zero-initialized, so its
    # weights moving away from zero proves the value function is being trained.
    # (Per-batch value loss / explained variance are too noisy under fresh-rollout
    # PPO at this scale to fall monotonically, so we assert the cleaner signal.)
    assert torch.any(trainer.policy.value_head.fc2.weight != 0.0)

    # Parameters actually moved (non-zero gradients were applied).
    assert any(m["train/gradient_norm"] > 0 for m in history)
    params_after = [p.detach() for p in trainer.policy.parameters() if p.requires_grad]
    assert any(not torch.allclose(a, b) for a, b in zip(params_before, params_after, strict=False))

    # The scheduler advanced exactly once per global step.
    assert trainer.scheduler.last_epoch == 10
    assert trainer.global_step == 10

    # The logger captured one record per step.
    assert trainer.logger is not None
    assert len(trainer.logger.history) == 10
