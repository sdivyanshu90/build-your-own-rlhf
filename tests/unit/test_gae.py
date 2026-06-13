"""Unit tests for Generalized Advantage Estimation (rollout.compute_gae)."""

from __future__ import annotations

import torch

from rlhf.training.ppo.rollout import compute_gae


def _reference_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    mask: torch.Tensor,
    gamma: float,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Canonical right-to-left GAE recursion (the implementation under test must match)."""
    b, t = rewards.shape
    adv = torch.zeros_like(rewards)
    for row in range(b):
        length = int(mask[row].sum().item())
        last_gae = 0.0
        for step in reversed(range(length)):
            next_value = float(values[row, step + 1]) if step + 1 < length else 0.0
            delta = float(rewards[row, step]) + gamma * next_value - float(values[row, step])
            last_gae = delta + gamma * lam * last_gae
            adv[row, step] = last_gae
    returns = (adv + values) * mask.to(values.dtype)
    return adv * mask.to(adv.dtype), returns


def test_gae_gamma1_lambda1_equals_monte_carlo() -> None:
    rewards = torch.randn(3, 6)
    values = torch.randn(3, 6)
    mask = torch.ones(3, 6, dtype=torch.bool)
    _, returns = compute_gae(rewards, values, mask, gamma=1.0, lam=1.0)
    # Monte-Carlo return R_t = sum_{j>=t} r_j (gamma = 1), independent of values.
    mc = torch.flip(torch.cumsum(torch.flip(rewards, dims=[1]), dim=1), dims=[1])
    assert torch.allclose(returns, mc, atol=1e-5)


def test_gae_gamma0_equals_immediate_rewards() -> None:
    rewards = torch.randn(2, 5)
    values = torch.randn(2, 5)
    mask = torch.ones(2, 5, dtype=torch.bool)
    _, returns = compute_gae(rewards, values, mask, gamma=0.0, lam=1.0)
    # With gamma = 0 the return collapses to the immediate reward.
    assert torch.allclose(returns, rewards, atol=1e-5)


def test_gae_respects_padding_mask() -> None:
    rewards = torch.randn(2, 6)
    values = torch.randn(2, 6)
    mask = torch.ones(2, 6, dtype=torch.bool)
    mask[0, 4:] = False  # row 0 has length 4
    mask[1, 5:] = False  # row 1 has length 5
    adv, returns = compute_gae(rewards, values, mask, gamma=0.99, lam=0.95)
    assert torch.all(adv[~mask] == 0.0)
    assert torch.all(returns[~mask] == 0.0)


def test_gae_matches_reference_loop() -> None:
    torch.manual_seed(0)
    rewards = torch.randn(4, 8)
    values = torch.randn(4, 8)
    mask = torch.ones(4, 8, dtype=torch.bool)
    mask[1, 6:] = False
    mask[2, 3:] = False
    for gamma, lam in [(1.0, 1.0), (0.99, 0.95), (0.95, 0.9), (0.0, 1.0)]:
        adv, returns = compute_gae(rewards, values, mask, gamma=gamma, lam=lam)
        ref_adv, ref_ret = _reference_gae(rewards, values, mask, gamma, lam)
        assert torch.allclose(adv, ref_adv, atol=1e-5), (gamma, lam)
        assert torch.allclose(returns, ref_ret, atol=1e-5), (gamma, lam)


def test_gae_handles_zero_length_sequences() -> None:
    rewards = torch.randn(2, 4)
    values = torch.randn(2, 4)
    mask = torch.ones(2, 4, dtype=torch.bool)
    mask[0] = False  # an entirely empty (zero-length) sequence
    adv, returns = compute_gae(rewards, values, mask, gamma=0.99, lam=0.95)
    assert torch.all(adv[0] == 0.0)
    assert torch.all(returns[0] == 0.0)
    assert torch.isfinite(adv).all()
    assert torch.isfinite(returns).all()
