"""Unit tests for the evaluator, human-eval export, and stopping criteria."""

from __future__ import annotations

import math

import pytest
import torch
from transformers import GPT2Config, PreTrainedTokenizerBase

from rlhf.evaluation.evaluator import EvalReport, Evaluator
from rlhf.evaluation.human_eval import export_for_human_eval
from rlhf.inference.stopping_criteria import StopOnSequences
from rlhf.models.policy import PolicyModel
from rlhf.models.reference_model import ReferenceModel
from rlhf.models.reward_model import RewardModel

PROMPTS = ["hello there", "the weather is", "once upon"]


def _setup(lm_config: GPT2Config) -> tuple[PolicyModel, ReferenceModel, RewardModel]:
    policy = PolicyModel.from_config(lm_config, value_head_dropout=0.0)
    reference = ReferenceModel.from_policy(policy)
    reward_model = RewardModel.from_config(lm_config, freeze_backbone=True)
    return policy, reference, reward_model


def test_evaluator_produces_finite_report(
    lm_config: GPT2Config, tokenizer: PreTrainedTokenizerBase
) -> None:
    policy, reference, reward_model = _setup(lm_config)
    evaluator = Evaluator(policy, reference, reward_model, tokenizer, max_new_tokens=6)
    report = evaluator.evaluate(PROMPTS)
    assert isinstance(report, EvalReport)
    assert math.isfinite(report.reward_mean)
    assert math.isfinite(report.kl_mean)
    assert report.perplexity >= 1.0 or math.isfinite(report.perplexity)
    assert report.num_prompts >= 1
    assert "mean" in report.response_length
    assert isinstance(report.to_dict(), dict)


def test_evaluator_win_rate_against_baseline(
    lm_config: GPT2Config, tokenizer: PreTrainedTokenizerBase
) -> None:
    from rlhf.inference.generation import BatchGenerator

    policy, reference, reward_model = _setup(lm_config)
    baseline = PolicyModel.from_config(lm_config, value_head_dropout=0.0)
    evaluator = Evaluator(policy, reference, reward_model, tokenizer, max_new_tokens=6)
    baseline_gen = BatchGenerator(baseline, tokenizer, max_new_tokens=6)
    report = evaluator.evaluate(PROMPTS, baseline_generator=baseline_gen)
    assert report.win_rate_vs_baseline is not None
    assert 0.0 <= report.win_rate_vs_baseline <= 1.0


def test_human_eval_export(
    lm_config: GPT2Config,
    tokenizer: PreTrainedTokenizerBase,
    tmp_path,  # type: ignore[no-untyped-def]
) -> None:
    policy, _, reward_model = _setup(lm_config)
    out = tmp_path / "human_eval.jsonl"
    records = export_for_human_eval(policy, reward_model, tokenizer, PROMPTS, out, max_new_tokens=6)
    assert len(records) == len(PROMPTS)
    assert out.is_file()
    assert all(math.isfinite(r.reward) for r in records)
    assert out.read_text().count("\n") == len(PROMPTS)


def test_human_eval_export_with_baseline(
    lm_config: GPT2Config,
    tokenizer: PreTrainedTokenizerBase,
    tmp_path,  # type: ignore[no-untyped-def]
) -> None:
    policy, _, reward_model = _setup(lm_config)
    baseline = PolicyModel.from_config(lm_config, value_head_dropout=0.0)
    records = export_for_human_eval(
        policy,
        reward_model,
        tokenizer,
        PROMPTS,
        tmp_path / "he.jsonl",
        max_new_tokens=6,
        baseline_policy=baseline,
    )
    assert any(r.baseline_response is not None for r in records)


def test_stop_on_sequences() -> None:
    criterion = StopOnSequences(stop_sequences=[[5, 6]], prompt_length=2)
    # Generated tail ends with the stop sequence for both rows -> stop.
    ids = torch.tensor([[0, 1, 9, 5, 6], [0, 1, 8, 5, 6]])
    assert criterion(ids, torch.zeros(2, 1)) is True
    # One row not yet terminated -> do not stop.
    ids2 = torch.tensor([[0, 1, 9, 5, 6], [0, 1, 8, 7, 7]])
    assert criterion(ids2, torch.zeros(2, 1)) is False


def test_stop_on_sequences_rejects_empty() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        StopOnSequences(stop_sequences=[], prompt_length=0)
