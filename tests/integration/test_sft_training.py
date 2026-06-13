"""Integration test: supervised fine-tuning (stage 0)."""

from __future__ import annotations

import math

from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerBase

from rlhf.config.schema import SFTConfig
from rlhf.training.sft.trainer import SFTTrainer

_EXAMPLES = [
    {"prompt": "Question: What is 2+2?\nAnswer:", "completion": " The answer is 4."},
    {"prompt": "Question: Capital of France?\nAnswer:", "completion": " It is Paris."},
    {"prompt": "Question: Color of the sky?\nAnswer:", "completion": " It is blue."},
    {"prompt": "Question: Largest planet?\nAnswer:", "completion": " It is Jupiter."},
]


def test_sft_training_reduces_loss(
    lm_config: GPT2Config,
    tokenizer: PreTrainedTokenizerBase,
) -> None:
    model = GPT2LMHeadModel(lm_config)
    config = SFTConfig(
        model_name_or_path="gpt2",
        learning_rate=1e-3,
        batch_size=2,
        epochs=5,
        max_length=64,
    )
    trainer = SFTTrainer(model, tokenizer, config)
    losses = trainer.train(_EXAMPLES * 2)  # 8 examples

    assert all(math.isfinite(loss) for loss in losses)
    # Training drives the next-token loss down over the run.
    assert sum(losses[-2:]) / 2 < sum(losses[:2]) / 2
    assert trainer.global_step == len(losses)
