"""
inference.generation — batched autoregressive sampling for PPO rollouts.

Overview
--------
:class:`BatchGenerator` wraps a :class:`~rlhf.models.policy.PolicyModel` and a
tokenizer to turn a batch of prompt strings into generated responses, using
left-padding (so continuations align) and the model's KV-cache (via HF
``generate``) for efficient incremental decoding.

It returns both the raw padded tensors and a trimmed per-example view
(:class:`RolloutSample`) that strips prompt left-padding and post-EOS padding —
exactly the ``(prompt_ids, response_ids)`` the PPO trainer re-scores and stores.

Legend: B = batch, P = prompt length, T = response length.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from transformers import PreTrainedTokenizerBase

from rlhf.data.preprocessing import build_generation_inputs
from rlhf.models.policy import PolicyModel


@dataclass
class RolloutSample:
    """A single trimmed rollout: real prompt ids, real response ids, decoded text."""

    prompt_ids: list[int]
    response_ids: list[int]
    response_text: str


@dataclass
class GeneratedBatch:
    """Raw batched generation output (tensors are left-padded on the prompt side)."""

    prompt_input_ids: Tensor
    prompt_attention_mask: Tensor
    sequences: Tensor
    response_ids: Tensor
    response_mask: Tensor
    sampling_logprobs: Tensor


class BatchGenerator:
    """
    Batched sampler over a policy model.

    Args:
        policy: The policy to sample from.
        tokenizer: Tokenizer for encoding prompts / decoding responses.
        max_new_tokens: Max tokens to generate per prompt.
        temperature: Sampling temperature.
        top_p: Nucleus sampling probability mass.
        top_k: Top-k truncation (0 disables).
        repetition_penalty: Repetition penalty applied during generation.
        max_prompt_length: Prompt truncation length.
        do_sample: Sample (True) vs greedy (False).
    """

    def __init__(
        self,
        policy: PolicyModel,
        tokenizer: PreTrainedTokenizerBase,
        *,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        repetition_penalty: float = 1.0,
        max_prompt_length: int = 256,
        do_sample: bool = True,
    ) -> None:
        self.policy = policy
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.max_prompt_length = max_prompt_length
        self.do_sample = do_sample

    @torch.no_grad()
    def generate(self, prompts: list[str]) -> GeneratedBatch:
        """Generate responses for a batch of prompt strings."""
        device = self.policy.device
        input_ids, attention = build_generation_inputs(
            self.tokenizer, prompts, max_length=self.max_prompt_length, device=device
        )
        out = self.policy.generate(
            input_ids,
            attention,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.do_sample,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            repetition_penalty=self.repetition_penalty,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        return GeneratedBatch(
            prompt_input_ids=input_ids,
            prompt_attention_mask=attention,
            sequences=out.sequences,
            response_ids=out.response_ids,
            response_mask=out.response_mask,
            sampling_logprobs=out.logprobs,
        )

    def to_samples(self, batch: GeneratedBatch) -> list[RolloutSample]:
        """
        Trim a :class:`GeneratedBatch` into per-example :class:`RolloutSample`s.

        Prompt left-padding and post-EOS response padding are removed so the
        returned ids are the real tokens only.
        """
        samples: list[RolloutSample] = []
        prompt_ids = batch.prompt_input_ids.cpu()
        prompt_mask = batch.prompt_attention_mask.cpu().bool()
        response_ids = batch.response_ids.cpu()
        response_mask = batch.response_mask.cpu().bool()
        for row in range(prompt_ids.shape[0]):
            real_prompt = prompt_ids[row][prompt_mask[row]].tolist()
            real_response = response_ids[row][response_mask[row]].tolist()
            text = self.tokenizer.decode(real_response, skip_special_tokens=True)
            samples.append(
                RolloutSample(
                    prompt_ids=real_prompt,
                    response_ids=real_response,
                    response_text=text,
                )
            )
        return samples


__all__ = ["BatchGenerator", "GeneratedBatch", "RolloutSample"]
