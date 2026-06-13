"""
evaluation.human_eval — export generations for human evaluation.

Overview
--------
Reward-model metrics are a proxy; periodically a human must look at samples.
:func:`export_for_human_eval` generates responses for a prompt set, scores them
with the reward model, and writes a JSON-lines file (one record per prompt) that
a labelling tool or spreadsheet can ingest. Optionally a baseline policy's
response is included side-by-side for pairwise preference collection.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from transformers import PreTrainedTokenizerBase

from rlhf.data.preprocessing import pad_sequences
from rlhf.inference.generation import BatchGenerator
from rlhf.models.policy import PolicyModel


@dataclass
class HumanEvalSample:
    """One row for human review."""

    prompt: str
    response: str
    reward: float
    baseline_response: str | None = None


def _score(reward_model: nn.Module, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    out = reward_model(ids, mask)
    reward: torch.Tensor = out[0] if isinstance(out, tuple) else out
    return reward


@torch.no_grad()
def export_for_human_eval(
    policy: PolicyModel,
    reward_model: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
    output_path: str | Path,
    *,
    max_new_tokens: int = 64,
    baseline_policy: PolicyModel | None = None,
    device: torch.device | None = None,
) -> list[HumanEvalSample]:
    """
    Generate, score, and export samples to a JSONL file for human evaluation.

    Args:
        policy: The policy under evaluation.
        reward_model: Reward model used to attach a scalar score per response.
        tokenizer: Tokenizer for encoding/decoding.
        prompts: Prompts to generate from.
        output_path: Destination ``.jsonl`` path.
        max_new_tokens: Generation length cap.
        baseline_policy: Optional baseline for side-by-side comparison.
        device: Compute device.

    Returns:
        The list of exported :class:`HumanEvalSample` records.
    """
    device = device or torch.device("cpu")
    pad_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else (tokenizer.eos_token_id or 0)
    )
    generator = BatchGenerator(policy, tokenizer, max_new_tokens=max_new_tokens, do_sample=True)
    batch = generator.generate(prompts)
    samples = generator.to_samples(batch)

    baseline_texts: list[str | None] = [None] * len(samples)
    if baseline_policy is not None:
        base_gen = BatchGenerator(
            baseline_policy, tokenizer, max_new_tokens=max_new_tokens, do_sample=True
        )
        base_samples = base_gen.to_samples(base_gen.generate(prompts))
        for i, bs in enumerate(base_samples[: len(samples)]):
            baseline_texts[i] = bs.response_text

    records: list[HumanEvalSample] = []
    for i, sample in enumerate(samples):
        if len(sample.response_ids) == 0:
            reward = 0.0
        else:
            ids, mask = pad_sequences([sample.prompt_ids + sample.response_ids], pad_id)
            reward = float(_score(reward_model, ids.to(device), mask.to(device))[0])
        records.append(
            HumanEvalSample(
                prompt=prompts[i] if i < len(prompts) else sample.response_text,
                response=sample.response_text,
                reward=reward,
                baseline_response=baseline_texts[i],
            )
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(asdict(record)) + "\n")
    return records


__all__ = ["HumanEvalSample", "export_for_human_eval"]
