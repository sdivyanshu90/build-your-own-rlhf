"""Unit tests for model base helpers, policy generation, and reference model."""

from __future__ import annotations

import pytest
import torch
from transformers import GPT2Config

from rlhf.exceptions import PolicyModelError
from rlhf.models.base import (
    last_token_indices,
    load_base_model,
    load_causal_lm,
    logprobs_from_logits,
    resolve_dtype,
)
from rlhf.models.policy import PolicyModel
from rlhf.models.reference_model import ReferenceModel


def test_resolve_dtype() -> None:
    assert resolve_dtype("float32") == torch.float32
    assert resolve_dtype("bfloat16") == torch.bfloat16
    assert resolve_dtype(None) is None
    assert resolve_dtype(torch.float16) == torch.float16
    with pytest.raises(ValueError, match="Unsupported dtype"):
        resolve_dtype("int4")


def test_logprobs_from_logits() -> None:
    logits = torch.randn(2, 3, 7)
    labels = torch.randint(0, 7, (2, 3))
    lp = logprobs_from_logits(logits, labels)
    assert lp.shape == (2, 3)
    expected = torch.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    assert torch.allclose(lp, expected)


def test_last_token_indices() -> None:
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0], [0, 0, 0, 0]])
    idx = last_token_indices(mask)
    assert idx.tolist() == [2, 1, 0]  # all-padding row clamps to 0


def test_load_helpers_require_exactly_one_source(tiny_config: GPT2Config) -> None:
    with pytest.raises(ValueError, match="exactly one"):
        load_causal_lm(None, config=None)
    with pytest.raises(ValueError, match="exactly one"):
        load_causal_lm("gpt2", config=tiny_config)
    with pytest.raises(ValueError, match="exactly one"):
        load_base_model(None, config=None)


def test_policy_requires_a_source() -> None:
    with pytest.raises(PolicyModelError):
        PolicyModel()


def test_policy_forward_and_score_shapes(tiny_config: GPT2Config) -> None:
    policy = PolicyModel.from_config(tiny_config, value_head_dropout=0.0)
    ids = torch.randint(3, 128, (2, 5))
    mask = torch.ones(2, 5, dtype=torch.long)
    logits, values = policy(ids, mask)
    assert logits.shape == (2, 5, tiny_config.vocab_size)
    assert values.shape == (2, 5)
    lp, vals, full_logits = policy.score_sequence(ids, mask)
    assert lp.shape == (2, 5)
    assert vals.shape == (2, 5)
    assert full_logits.shape == (2, 5, tiny_config.vocab_size)
    # Position 0 carries no predicted log-prob / value (nothing precedes it).
    assert torch.all(lp[:, 0] == 0.0)


def test_policy_generate_and_mask(tiny_config: GPT2Config) -> None:
    policy = PolicyModel.from_config(tiny_config, value_head_dropout=0.0)
    prompt = torch.randint(3, 128, (2, 3))
    pmask = torch.ones(2, 3, dtype=torch.long)
    out = policy.generate(prompt, pmask, max_new_tokens=5, do_sample=True)
    assert out.response_ids.shape[0] == 2
    assert out.logprobs.shape == out.response_ids.shape
    assert out.response_mask.shape == out.response_ids.shape
    assert out.prompt_length == 3


def test_policy_freeze_layers(tiny_config: GPT2Config) -> None:
    policy = PolicyModel.from_config(tiny_config, freeze_layers=1)
    # The first transformer block's parameters are frozen.
    block0 = policy.backbone.transformer.h[0]
    assert all(not p.requires_grad for p in block0.parameters())
    # The value head remains trainable.
    assert any(p.requires_grad for p in policy.value_head.parameters())


def test_policy_save_and_load_value_head(tiny_config: GPT2Config, tmp_path) -> None:  # type: ignore[no-untyped-def]
    policy = PolicyModel.from_config(tiny_config, value_head_dropout=0.0)
    # Perturb the value head so save/load round-trips a non-trivial state.
    with torch.no_grad():
        policy.value_head.fc2.weight.add_(1.0)
    policy.save_pretrained(tmp_path / "policy")
    fresh = PolicyModel.from_config(tiny_config, value_head_dropout=0.0)
    fresh.load_value_head(tmp_path / "policy")
    assert torch.allclose(fresh.value_head.fc2.weight, policy.value_head.fc2.weight)


def test_reference_model_logprobs_and_frozen(tiny_config: GPT2Config) -> None:
    ref = ReferenceModel.from_config(tiny_config)
    assert all(not p.requires_grad for p in ref.parameters())
    ids = torch.randint(3, 128, (2, 4))
    mask = torch.ones(2, 4, dtype=torch.long)
    lp = ref.compute_logprobs(ids, mask)
    assert lp.shape == (2, 4)
    assert torch.all(lp[:, 0] == 0.0)


def test_reference_from_policy_requires_backbone() -> None:
    with pytest.raises(PolicyModelError):
        ReferenceModel.from_policy(torch.nn.Linear(2, 2))
