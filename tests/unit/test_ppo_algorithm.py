"""Unit tests for the core PPO loss (algorithm.compute_ppo_loss)."""

from __future__ import annotations

import math

import pytest
import torch

from rlhf.training.ppo.algorithm import compute_ppo_loss

B, T, V = 3, 5, 11
_DEFAULTS = {
    "clip_eps": 0.2,
    "clip_eps_vf": 0.2,
    "entropy_coeff": 0.01,
    "value_coeff": 0.5,
}


def _call(**overrides: object):  # type: ignore[no-untyped-def]
    """Invoke compute_ppo_loss with sensible defaults, overriding selected args."""
    kwargs: dict[str, object] = {
        "logprobs": torch.zeros(B, T),
        "old_logprobs": torch.zeros(B, T),
        "advantages": torch.ones(B, T),
        "returns": torch.zeros(B, T),
        "values": torch.zeros(B, T),
        "old_values": torch.zeros(B, T),
        "mask": torch.ones(B, T, dtype=torch.bool),
        "vocab_logits": torch.randn(B, T, V),
        **_DEFAULTS,
    }
    kwargs.update(overrides)
    return compute_ppo_loss(**kwargs)  # type: ignore[arg-type]


def test_policy_loss_negative_for_positive_advantages() -> None:
    # ratio == 1 (logprobs == old) and A > 0 => policy_loss = -mean(A) < 0.
    out = _call(advantages=torch.ones(B, T))
    assert out.policy_loss <= 0.0


@pytest.mark.parametrize("ratio", [1.5, 0.5])
def test_clipping_activates_outside_trust_region(ratio: float) -> None:
    # Set logprobs so that exp(logprobs - old) == ratio, which is outside
    # [1-eps, 1+eps] = [0.8, 1.2] for both parametrized values.
    logprobs = torch.full((B, T), math.log(ratio))
    out = _call(logprobs=logprobs, old_logprobs=torch.zeros(B, T))
    assert out.clip_fraction == pytest.approx(1.0)


def test_clip_fraction_zero_when_ratio_one() -> None:
    out = _call(logprobs=torch.zeros(B, T), old_logprobs=torch.zeros(B, T))
    assert out.clip_fraction == pytest.approx(0.0)


def test_approx_kl_zero_when_logprobs_equal() -> None:
    lp = torch.randn(B, T)
    out = _call(logprobs=lp, old_logprobs=lp.clone())
    assert out.approx_kl == pytest.approx(0.0, abs=1e-6)


def test_explained_variance_one_when_values_equal_returns() -> None:
    returns = torch.randn(B, T)
    out = _call(returns=returns, values=returns.clone())
    assert out.explained_variance == pytest.approx(1.0, abs=1e-5)


def test_explained_variance_zero_when_values_constant() -> None:
    returns = torch.randn(B, T)
    out = _call(returns=returns, values=torch.full((B, T), 3.0))
    assert out.explained_variance == pytest.approx(0.0, abs=1e-5)


def test_value_loss_clipping_uses_pessimistic_max() -> None:
    old_values = torch.zeros(B, T)
    returns = torch.zeros(B, T)
    # Push predictions well beyond the value clip range; the unclipped squared
    # error is then the larger (pessimistic) term that the loss must select.
    values = torch.full((B, T), 5.0)
    out = _call(values=values, old_values=old_values, returns=returns, clip_eps_vf=0.2)
    clipped_pred = 0.0 + 0.2  # old + clip range
    expected = max((5.0 - 0.0) ** 2, (clipped_pred - 0.0) ** 2)
    assert out.value_loss == pytest.approx(expected, rel=1e-5)


def test_entropy_reduces_total_loss() -> None:
    logits = torch.randn(B, T, V)
    with_entropy = _call(vocab_logits=logits, entropy_coeff=0.1)
    without_entropy = _call(vocab_logits=logits, entropy_coeff=0.0)
    assert float(with_entropy.total_loss) < float(without_entropy.total_loss)
    assert with_entropy.entropy_loss > 0.0


def test_mask_excludes_padding_positions() -> None:
    mask = torch.ones(B, T, dtype=torch.bool)
    mask[:, T - 2 :] = False
    # Share a single vocab_logits tensor so the entropy term is identical across
    # both calls and only the (masked) advantage difference is under test.
    logits = torch.randn(B, T, V)
    base = _call(mask=mask, vocab_logits=logits)
    # Inject extreme garbage into the padded positions; the masked loss must not
    # change because those positions are excluded from every reduction.
    poisoned_adv = torch.ones(B, T)
    poisoned_adv[:, T - 2 :] = 1e6
    poisoned = _call(mask=mask, advantages=poisoned_adv, vocab_logits=logits)
    assert float(base.total_loss) == pytest.approx(float(poisoned.total_loss), rel=1e-6)


def test_all_padding_batch_is_zero_not_nan() -> None:
    out = _call(mask=torch.zeros(B, T, dtype=torch.bool))
    assert float(out.total_loss) == pytest.approx(0.0)
    assert math.isfinite(float(out.total_loss))


def test_loss_is_finite_and_differentiable() -> None:
    logprobs = torch.zeros(B, T, requires_grad=True)
    values = torch.zeros(B, T, requires_grad=True)
    vocab_logits = torch.randn(B, T, V, requires_grad=True)
    out = compute_ppo_loss(
        logprobs=logprobs,
        old_logprobs=torch.zeros(B, T),
        advantages=torch.randn(B, T),
        returns=torch.randn(B, T),
        values=values,
        old_values=torch.zeros(B, T),
        mask=torch.ones(B, T, dtype=torch.bool),
        vocab_logits=vocab_logits,
        **_DEFAULTS,
    )
    assert torch.isfinite(out.total_loss)
    out.total_loss.backward()
    assert logprobs.grad is not None
    assert values.grad is not None
    assert vocab_logits.grad is not None
