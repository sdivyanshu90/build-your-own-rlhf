"""Unit tests for the reward model, Bradley-Terry loss, and ensemble."""

from __future__ import annotations

import math

import pytest
import torch
from transformers import GPT2Config

from rlhf.models.reward_model import (
    RewardModel,
    RewardModelEnsemble,
    RunningMoments,
    bradley_terry_loss,
)


def test_bradley_terry_loss_log2_when_equal() -> None:
    r = torch.randn(8)
    loss = bradley_terry_loss(r, r.clone())
    assert float(loss) == pytest.approx(math.log(2), abs=1e-6)


def test_bradley_terry_loss_decreases_with_margin() -> None:
    chosen = torch.ones(8)
    rejected = -torch.ones(8)
    assert float(bradley_terry_loss(chosen, rejected)) < math.log(2)


def test_gradient_flows_through_unfrozen_backbone(tiny_config: GPT2Config) -> None:
    rm = RewardModel.from_config(tiny_config, freeze_backbone=False)
    # Chosen and rejected must be *different* sequences, else the reward terms
    # cancel in (r_chosen - r_rejected) and the gradient is identically zero.
    ids_chosen = torch.randint(3, 128, (4, 6))
    ids_rejected = torch.randint(3, 128, (4, 6))
    mask = torch.ones(4, 6, dtype=torch.long)
    bradley_terry_loss(rm(ids_chosen, mask), rm(ids_rejected, mask)).backward()
    grads = [p.grad for p in rm.backbone.parameters() if p.requires_grad]
    assert any(g is not None and torch.any(g != 0) for g in grads)


def test_no_gradient_through_frozen_backbone(tiny_config: GPT2Config) -> None:
    rm = RewardModel.from_config(tiny_config, freeze_backbone=True)
    ids = torch.randint(3, 128, (4, 6))
    mask = torch.ones(4, 6, dtype=torch.long)
    rm(ids, mask).sum().backward()
    assert all(p.grad is None for p in rm.backbone.parameters())
    # The reward head still receives gradient.
    assert rm.reward_head.weight.grad is not None


def test_running_moments_match_reference() -> None:
    moments = RunningMoments()
    torch.manual_seed(0)
    batch_a = torch.randn(50)
    batch_b = torch.randn(37)
    moments.update(batch_a)
    moments.update(batch_b)
    full = torch.cat([batch_a, batch_b])
    assert float(moments.mean) == pytest.approx(float(full.mean()), abs=1e-5)
    assert float(moments.variance) == pytest.approx(float(full.var(unbiased=False)), abs=1e-5)


def test_normalizer_is_identity_before_data() -> None:
    moments = RunningMoments()
    x = torch.randn(5)
    assert torch.allclose(moments.normalize(x), x)


def test_ensemble_returns_mean_and_std(tiny_config: GPT2Config) -> None:
    ensemble = RewardModelEnsemble.from_config(tiny_config, ensemble_size=3, freeze_backbone=False)
    ids = torch.randint(3, 128, (5, 6))
    mask = torch.ones(5, 6, dtype=torch.long)
    mean, std = ensemble(ids, mask)
    assert mean.shape == (5,)
    assert std.shape == (5,)
    assert torch.all(std >= 0)
    # Manually reproduce mean across members.
    stacked = torch.stack([m(ids, mask) for m in ensemble.models], dim=0)
    assert torch.allclose(mean, stacked.mean(dim=0), atol=1e-5)


def test_single_member_ensemble_has_zero_std(tiny_config: GPT2Config) -> None:
    ensemble = RewardModelEnsemble.from_config(tiny_config, ensemble_size=1)
    ids = torch.randint(3, 128, (3, 5))
    mask = torch.ones(3, 5, dtype=torch.long)
    _, std = ensemble(ids, mask)
    assert torch.allclose(std, torch.zeros_like(std))


def test_reward_pooled_at_last_non_padding_token(tiny_config: GPT2Config) -> None:
    rm = RewardModel.from_config(tiny_config, freeze_backbone=True)
    rm.eval()
    tokens = torch.tensor([[5, 9, 11]])
    # Same content, but right-padded with two extra (masked) tokens.
    padded = torch.tensor([[5, 9, 11, 0, 0]])
    mask_short = torch.ones(1, 3, dtype=torch.long)
    mask_padded = torch.tensor([[1, 1, 1, 0, 0]])
    r_short = rm(tokens, mask_short)
    r_padded = rm(padded, mask_padded)
    # Pooling at the last real token must ignore the padding entirely.
    assert torch.allclose(r_short, r_padded, atol=1e-5)
