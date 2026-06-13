"""
training.sft.trainer — supervised fine-tuning (RLHF stage 0).

Overview
--------
:class:`SFTTrainer` fine-tunes a causal LM on ``(prompt, completion)`` pairs with
the standard next-token cross-entropy, masking the prompt tokens out of the loss
so the model is trained only to produce the completion. The resulting model is
the starting point for both the PPO policy and the frozen reference.

Mathematical Background
-----------------------
Loss is mean token-level negative log-likelihood over completion positions::

    L_SFT = -(1/|C|) * sum_{t in C} log p_theta(x_t | x_<t)

where ``C`` is the set of completion token positions.

Usage Example
-------------
>>> # trainer = SFTTrainer(model, tokenizer, SFTConfig(model_name_or_path="gpt2"))
>>> # history = trainer.train([{"prompt": "Q: hi\\n", "completion": "A: hello"}])
"""

from __future__ import annotations

import logging

import torch
from torch import Tensor, nn
from torch.optim import AdamW
from transformers import PreTrainedTokenizerBase

from rlhf.config.schema import SFTConfig
from rlhf.exceptions import DataValidationError
from rlhf.monitoring.logger import RLHFLogger
from rlhf.training.ppo.scheduler import build_lr_scheduler
from rlhf.utils import set_seed

logger = logging.getLogger(__name__)

# Ignore index used by cross-entropy for masked (prompt / padding) positions.
_IGNORE_INDEX: int = -100


class SFTTrainer:
    """Supervised fine-tuning trainer for a causal LM."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        config: SFTConfig,
        device: torch.device | None = None,
        logger_backend: RLHFLogger | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device or torch.device("cpu")
        self.model.to(self.device)
        self.logger = logger_backend
        self.pad_id = (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else (tokenizer.eos_token_id or 0)
        )
        self.optimizer = AdamW(self.model.parameters(), lr=config.learning_rate)
        # A constant default so train_step() works even if called standalone;
        # train() replaces this with the real warmup+decay schedule.
        self.scheduler = build_lr_scheduler(self.optimizer, "constant", 0, 1)
        self.global_step = 0

    def _encode_example(self, prompt: str, completion: str) -> tuple[list[int], int]:
        """Encode one example, returning the full id list and the prompt length."""
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        completion_ids = self.tokenizer.encode(completion, add_special_tokens=False)
        if self.tokenizer.eos_token_id is not None:
            completion_ids = [*completion_ids, self.tokenizer.eos_token_id]
        full = (prompt_ids + completion_ids)[: self.config.max_length]
        prompt_len = min(len(prompt_ids), len(full))
        return full, prompt_len

    def _build_batch(self, examples: list[dict[str, str]]) -> tuple[Tensor, Tensor, Tensor]:
        """Tokenize, pad, and build label tensors with the prompt masked out."""
        encoded = [self._encode_example(ex["prompt"], ex["completion"]) for ex in examples]
        max_len = max(len(ids) for ids, _ in encoded)
        n = len(encoded)
        input_ids = torch.full((n, max_len), self.pad_id, dtype=torch.long)
        attention = torch.zeros(n, max_len, dtype=torch.long)
        labels = torch.full((n, max_len), _IGNORE_INDEX, dtype=torch.long)
        for i, (ids, prompt_len) in enumerate(encoded):
            length = len(ids)
            input_ids[i, :length] = torch.tensor(ids, dtype=torch.long)
            attention[i, :length] = 1
            # Supervise only completion tokens: mask prompt + padding to ignore.
            if length > prompt_len:
                labels[i, prompt_len:length] = torch.tensor(
                    ids[prompt_len:length], dtype=torch.long
                )
        return (
            input_ids.to(self.device),
            attention.to(self.device),
            labels.to(self.device),
        )

    def train_step(self, batch: tuple[Tensor, Tensor, Tensor]) -> float:
        """Run one optimization step on a pre-built batch; returns the loss."""
        input_ids, attention, labels = batch
        self.model.train()
        outputs = self.model(input_ids=input_ids, attention_mask=attention, labels=labels)
        loss = outputs.loss
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
        self.scheduler.step()
        self.global_step += 1
        return float(loss.detach())

    def train(self, examples: list[dict[str, str]]) -> list[float]:
        """
        Fine-tune on ``examples`` for ``config.epochs`` epochs.

        Args:
            examples: List of ``{"prompt": ..., "completion": ...}`` dicts.

        Returns:
            The list of per-step losses.
        """
        if not examples:
            raise DataValidationError("SFT requires at least one example.")
        set_seed(self.config.seed)
        steps_per_epoch = (len(examples) + self.config.batch_size - 1) // self.config.batch_size
        total_steps = max(steps_per_epoch * self.config.epochs, 1)
        self.scheduler = build_lr_scheduler(
            self.optimizer, "linear", self.config.warmup_steps, total_steps
        )
        losses: list[float] = []
        for epoch in range(self.config.epochs):
            for start in range(0, len(examples), self.config.batch_size):
                chunk = examples[start : start + self.config.batch_size]
                loss = self.train_step(self._build_batch(chunk))
                losses.append(loss)
                if self.logger is not None:
                    self.logger.log_ppo_step(self.global_step, {"sft/loss": loss})
            logger.info("SFT epoch %d complete (last loss %.4f)", epoch, losses[-1])
        return losses


__all__ = ["SFTTrainer"]
