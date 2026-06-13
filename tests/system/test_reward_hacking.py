"""System test: reward over-optimization is detected and the last checkpoint is safe."""

from __future__ import annotations

import pytest
import torch
from transformers import GPT2Config, PreTrainedTokenizerBase

from rlhf.config.schema import PPOConfig
from rlhf.exceptions import RewardHackingDetected
from rlhf.models.policy import PolicyModel
from rlhf.models.reference_model import ReferenceModel
from rlhf.security.audit import CheckpointVerifier
from rlhf.training.ppo.trainer import PPOTrainer

PROMPTS = ["alpha", "beta", "gamma", "delta"]


def _evil_reward(
    input_ids: torch.Tensor, attention_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """A broken reward model that always returns a saturated reward of +100."""
    n = input_ids.shape[0]
    return torch.full((n,), 100.0), torch.zeros(n)


def test_reward_hacking_detected_and_checkpoint_preserved(
    lm_config: GPT2Config,
    tokenizer: PreTrainedTokenizerBase,
    tmp_path,  # type: ignore[no-untyped-def]
) -> None:
    policy = PolicyModel.from_config(lm_config, value_head_dropout=0.0)
    reference = ReferenceModel.from_policy(policy)
    config = PPOConfig(
        rollout_batch_size=4,
        mini_batch_size=2,
        ppo_epochs=1,
        max_new_tokens=6,
        warmup_steps=0,
        total_steps=10,
        reward_hacking_threshold=0.2,
    )
    trainer = PPOTrainer(
        policy,
        reference,
        tokenizer,
        config,
        PROMPTS,
        reward_scorer=_evil_reward,
        output_dir=str(tmp_path),
    )

    # Save the last-known-good checkpoint with an integrity manifest BEFORE the
    # diverging optimization begins.
    verifier = CheckpointVerifier()
    ckpt_path = tmp_path / "pre_hack.pt"
    digest = verifier.save(trainer.state_dict(), ckpt_path, step=0)

    # Training must abort with RewardHackingDetected before reaching step 10.
    with pytest.raises(RewardHackingDetected):
        trainer.train(num_steps=10)
    assert trainer.global_step < 10

    # The checkpoint taken prior to detection is intact and verifiably loadable.
    restored = verifier.load_verified(ckpt_path, digest)
    assert "policy_backbone" in restored
    assert restored["global_step"] == 0
