"""System test: SFT -> reward model -> PPO end-to-end on toy data."""

from __future__ import annotations

from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerBase

from rlhf.config.schema import PPOConfig, RewardModelConfig, SFTConfig
from rlhf.data.schemas import Preference
from rlhf.models.policy import PolicyModel
from rlhf.models.reference_model import ReferenceModel
from rlhf.models.reward_model import RewardModel
from rlhf.training.ppo.trainer import PPOTrainer
from rlhf.training.reward_model.trainer import RewardModelTrainer
from rlhf.training.sft.trainer import SFTTrainer

PROMPTS = [
    "Question: greeting?\nAnswer:",
    "Question: weather?\nAnswer:",
    "Question: a fact?\nAnswer:",
    "Question: a number?\nAnswer:",
    "Question: a color?\nAnswer:",
    "Question: a place?\nAnswer:",
]
SFT_EXAMPLES = [{"prompt": p, "completion": " a helpful answer."} for p in PROMPTS]


def test_full_pipeline_sft_rm_ppo(
    lm_config: GPT2Config,
    tokenizer: PreTrainedTokenizerBase,
    dummy_preferences: list[Preference],
    tmp_path,  # type: ignore[no-untyped-def]
) -> None:
    # --- Stage 0: SFT ---
    base = GPT2LMHeadModel(lm_config)
    sft_config = SFTConfig(model_name_or_path="gpt2", learning_rate=1e-3, batch_size=3, epochs=3)
    SFTTrainer(base, tokenizer, sft_config).train(SFT_EXAMPLES)

    # --- Stage 1: reward model ---
    reward_model = RewardModel.from_config(lm_config, freeze_backbone=False)
    rm_config = RewardModelConfig(
        model_name_or_path="gpt2",
        freeze_backbone=False,
        learning_rate=1e-3,
        batch_size=16,
        epochs=3,
    )
    RewardModelTrainer(reward_model, tokenizer, rm_config).train(dummy_preferences[:60])

    # --- Stage 2: PPO (policy initialized from the SFT model) ---
    policy = PolicyModel(backbone=base, value_head_dropout=0.0)
    reference = ReferenceModel.from_policy(policy)
    ppo_config = PPOConfig(
        rollout_batch_size=6,
        mini_batch_size=3,
        ppo_epochs=2,
        max_new_tokens=8,
        learning_rate=5e-4,
        warmup_steps=0,
        total_steps=20,
        save_every=1000,
        reward_hacking_threshold=5.0,  # permissive: focus on reward/KL behaviour
    )
    trainer = PPOTrainer(
        policy,
        reference,
        tokenizer,
        ppo_config,
        PROMPTS,
        reward_model=reward_model,
        output_dir=str(tmp_path),
        reward_saturation_ceiling=1e9,
    )
    history = trainer.train(num_steps=20)

    # KL from the reference stays bounded throughout.
    assert all(m["train/kl_divergence"] < 5.0 for m in history)

    # PPO increases the reward the policy earns (late window beats early window).
    early = sum(m["reward/mean"] for m in history[:5]) / 5
    late = sum(m["reward/mean"] for m in history[-5:]) / 5
    assert late > early

    # A checkpoint is saved and reloads cleanly.
    ckpt = trainer.save_checkpoint(tmp_path / "final")
    assert ckpt.is_file()
    fresh = PPOTrainer(
        PolicyModel(backbone=GPT2LMHeadModel(lm_config), value_head_dropout=0.0),
        ReferenceModel.from_config(lm_config),
        tokenizer,
        ppo_config,
        PROMPTS,
        reward_model=reward_model,
        output_dir=str(tmp_path),
    )
    fresh.load_checkpoint(tmp_path / "final")
    assert fresh.global_step == trainer.global_step
