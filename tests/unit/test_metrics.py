"""Unit tests for evaluation.metrics."""

from __future__ import annotations

import math

import pytest
import torch

from rlhf.evaluation import metrics


def test_all_metrics_return_finite_floats() -> None:
    mask = torch.ones(2, 4, dtype=torch.bool)
    lp = torch.log_softmax(torch.randn(2, 4, 6), dim=-1)
    ref = torch.log_softmax(torch.randn(2, 4, 6), dim=-1)
    values = [
        metrics.kl_divergence(lp, ref, mask),
        metrics.approx_kl(torch.randn(2, 4), torch.randn(2, 4), mask),
        metrics.explained_variance(torch.randn(8), torch.randn(8)),
        metrics.clip_fraction(torch.rand(8) + 0.5, 0.2),
        metrics.reward_win_rate(torch.randn(8), torch.randn(8)),
        metrics.reward_hacking_score(torch.randn(8).abs() + 1, torch.rand(8)),
        metrics.perplexity(-torch.rand(8), torch.tensor([4, 4])),
    ]
    for v in values:
        assert isinstance(v, float)
        assert math.isfinite(v)
    stats = metrics.response_length_stats(torch.tensor([3, 5, 7, 9]))
    assert all(math.isfinite(x) for x in stats.values())


def test_kl_divergence_zero_when_equal() -> None:
    # 3-D full-distribution form.
    lp = torch.log_softmax(torch.randn(2, 3, 5), dim=-1)
    mask = torch.ones(2, 3, dtype=torch.bool)
    assert metrics.kl_divergence(lp, lp.clone(), mask) == pytest.approx(0.0, abs=1e-6)
    # 2-D token-level form.
    tok = torch.randn(2, 3)
    assert metrics.kl_divergence(tok, tok.clone(), mask) == pytest.approx(0.0, abs=1e-6)


def test_kl_divergence_matches_torch_distributions() -> None:
    logits_p = torch.randn(1, 1, 7)
    logits_q = torch.randn(1, 1, 7)
    lp = torch.log_softmax(logits_p, dim=-1)
    lq = torch.log_softmax(logits_q, dim=-1)
    mask = torch.ones(1, 1, dtype=torch.bool)
    p = torch.distributions.Categorical(logits=logits_p.squeeze())
    q = torch.distributions.Categorical(logits=logits_q.squeeze())
    expected = float(torch.distributions.kl_divergence(p, q))
    assert metrics.kl_divergence(lp, lq, mask) == pytest.approx(expected, abs=1e-5)


def test_explained_variance_edge_cases() -> None:
    x = torch.randn(16)
    assert metrics.explained_variance(x, x.clone()) == pytest.approx(1.0, abs=1e-5)
    const = torch.full((16,), 2.0)
    assert metrics.explained_variance(const, x) == pytest.approx(0.0, abs=1e-5)
    # Zero-variance returns must not divide-by-zero.
    assert metrics.explained_variance(x, const) == 0.0


def test_clip_fraction_basic() -> None:
    assert metrics.clip_fraction(torch.ones(10), 0.2) == pytest.approx(0.0)
    assert metrics.clip_fraction(torch.full((10,), 2.0), 0.2) == pytest.approx(1.0)


def test_reward_win_rate_half_when_equal() -> None:
    r = torch.randn(20)
    assert metrics.reward_win_rate(r, r.clone()) == pytest.approx(0.5)
    assert metrics.reward_win_rate(r + 1, r) == pytest.approx(1.0)
    assert metrics.reward_win_rate(r, r + 1) == pytest.approx(0.0)


def test_response_length_stats_values() -> None:
    stats = metrics.response_length_stats(torch.tensor([10, 20, 30, 40, 50]))
    assert stats["mean"] == pytest.approx(30.0)
    assert stats["p50"] == pytest.approx(30.0)
    assert stats["p95"] <= 50.0


def test_perplexity_uniform() -> None:
    vocab = 10
    # A uniform distribution over V tokens has perplexity exactly V.
    logprobs = torch.full((100,), -math.log(vocab))
    assert metrics.perplexity(logprobs) == pytest.approx(vocab, rel=1e-4)


def test_metrics_handle_empty_inputs() -> None:
    empty_mask = torch.zeros(2, 3, dtype=torch.bool)
    assert metrics.kl_divergence(torch.randn(2, 3), torch.randn(2, 3), empty_mask) == 0.0
    assert metrics.approx_kl(torch.randn(2, 3), torch.randn(2, 3), empty_mask) == 0.0
    assert metrics.explained_variance(torch.tensor([]), torch.tensor([])) == 0.0
    assert metrics.clip_fraction(torch.tensor([]), 0.2) == 0.0
    assert metrics.reward_win_rate(torch.tensor([]), torch.tensor([])) == 0.5
    assert metrics.reward_hacking_score(torch.tensor([]), torch.tensor([])) == 0.0
    assert metrics.perplexity(torch.tensor([])) == 1.0
    assert metrics.response_length_stats(torch.tensor([]))["mean"] == 0.0
