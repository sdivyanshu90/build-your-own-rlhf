"""
evaluation.evaluator — offline evaluation of an RLHF policy.

Overview
--------
:class:`Evaluator` runs a held-out prompt set through the policy and reports the
metrics that matter for RLHF: mean reward, KL from the reference, perplexity,
response-length statistics, and (optionally) the win-rate of the policy's
responses against a baseline policy's responses under the reward model.

Usage Example
-------------
>>> # ev = Evaluator(policy, reference, reward_model, tokenizer)
>>> # report = ev.evaluate(prompts)
>>> # report.reward_mean, report.kl_mean, report.perplexity
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import torch
from torch import nn
from transformers import PreTrainedTokenizerBase

from rlhf.data.preprocessing import pad_sequences
from rlhf.evaluation import metrics
from rlhf.inference.generation import BatchGenerator
from rlhf.models.policy import PolicyModel
from rlhf.models.reference_model import ReferenceModel


@dataclass
class EvalReport:
    """Aggregate evaluation metrics for one held-out prompt set."""

    reward_mean: float
    reward_std: float
    kl_mean: float
    perplexity: float
    response_length: dict[str, float] = field(default_factory=dict)
    win_rate_vs_baseline: float | None = None
    num_prompts: int = 0

    def to_dict(self) -> dict[str, object]:
        """Flatten the report to a JSON-serializable dict."""
        return asdict(self)


class Evaluator:
    """Evaluates a policy on a held-out prompt set."""

    def __init__(
        self,
        policy: PolicyModel,
        reference: ReferenceModel,
        reward_model: nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        *,
        max_new_tokens: int = 64,
        device: torch.device | None = None,
    ) -> None:
        self.policy = policy
        self.reference = reference
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.device = device or torch.device("cpu")
        self.pad_id = (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else (tokenizer.eos_token_id or 0)
        )
        self.generator = BatchGenerator(
            policy, tokenizer, max_new_tokens=max_new_tokens, do_sample=True
        )

    def _reward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.reward_model(input_ids, attention_mask)
        reward: torch.Tensor = out[0] if isinstance(out, tuple) else out
        return reward

    @torch.no_grad()
    def evaluate(
        self, prompts: list[str], baseline_generator: BatchGenerator | None = None
    ) -> EvalReport:
        """
        Evaluate the policy on ``prompts``.

        Args:
            prompts: Held-out evaluation prompts.
            baseline_generator: Optional generator (e.g. wrapping the SFT model)
                used to compute a paired win-rate.

        Returns:
            An :class:`EvalReport`.
        """
        self.policy.eval()
        batch = self.generator.generate(prompts)
        samples = self.generator.to_samples(batch)
        samples = [s for s in samples if len(s.response_ids) > 0]
        if not samples:
            return EvalReport(0.0, 0.0, 0.0, 1.0, num_prompts=len(prompts))

        full_seqs = [s.prompt_ids + s.response_ids for s in samples]
        full_ids, full_mask = pad_sequences(full_seqs, self.pad_id)
        full_ids = full_ids.to(self.device)
        full_mask = full_mask.to(self.device)

        # Build a response mask aligned to the full sequence for KL / perplexity.
        response_mask = torch.zeros_like(full_mask, dtype=torch.bool)
        for i, s in enumerate(samples):
            p_len, r_len = len(s.prompt_ids), len(s.response_ids)
            response_mask[i, p_len : p_len + r_len] = True

        policy_logprobs, _, _ = self.policy.score_sequence(full_ids, full_mask)
        ref_logprobs = self.reference.compute_logprobs(full_ids, full_mask)
        rewards = self._reward(full_ids, full_mask)

        kl_mean = metrics.kl_divergence(policy_logprobs, ref_logprobs, response_mask)
        flat_logprobs = policy_logprobs[response_mask]
        ppl = metrics.perplexity(flat_logprobs)
        lengths = torch.tensor([len(s.response_ids) for s in samples])

        win_rate: float | None = None
        if baseline_generator is not None:
            win_rate = self._win_rate(prompts, rewards, baseline_generator)

        return EvalReport(
            reward_mean=float(rewards.mean()),
            reward_std=float(rewards.std(unbiased=False)),
            kl_mean=kl_mean,
            perplexity=ppl,
            response_length=metrics.response_length_stats(lengths),
            win_rate_vs_baseline=win_rate,
            num_prompts=len(samples),
        )

    @torch.no_grad()
    def _win_rate(
        self,
        prompts: list[str],
        policy_rewards: torch.Tensor,
        baseline_generator: BatchGenerator,
    ) -> float:
        """Paired win-rate: policy reward vs baseline reward on the same prompts."""
        base_batch = baseline_generator.generate(prompts)
        base_samples = baseline_generator.to_samples(base_batch)
        base_samples = [s for s in base_samples if len(s.response_ids) > 0]
        if not base_samples:
            return 0.5
        full = [s.prompt_ids + s.response_ids for s in base_samples]
        ids, mask = pad_sequences(full, self.pad_id)
        base_rewards = self._reward(ids.to(self.device), mask.to(self.device))
        n = min(policy_rewards.shape[0], base_rewards.shape[0])
        return metrics.reward_win_rate(policy_rewards[:n], base_rewards[:n])


__all__ = ["EvalReport", "Evaluator"]
