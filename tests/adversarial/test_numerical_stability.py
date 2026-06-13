"""Adversarial tests: NaN/Inf propagation and extreme-input numerical stability."""

from __future__ import annotations

import pytest
import torch

from rlhf.evaluation.metrics import explained_variance
from rlhf.training.ppo.algorithm import compute_ppo_loss
from rlhf.training.ppo.rollout import compute_gae

B, T, V = 2, 4, 8
_DEFAULTS = {
    "clip_eps": 0.2,
    "clip_eps_vf": 0.2,
    "entropy_coeff": 0.01,
    "value_coeff": 0.5,
}

# float16 matmul/log_softmax are not implemented for many CPU ops, so skip it on CPU.
_DTYPES = [torch.float32, torch.bfloat16, torch.float16]


def _supported(dtype: torch.dtype) -> bool:
    return not (dtype == torch.float16 and not torch.cuda.is_available())


def _loss(dtype: torch.dtype, **overrides: object):  # type: ignore[no-untyped-def]
    kwargs: dict[str, object] = {
        "logprobs": torch.zeros(B, T, dtype=dtype),
        "old_logprobs": torch.zeros(B, T, dtype=dtype),
        "advantages": torch.ones(B, T, dtype=dtype),
        "returns": torch.zeros(B, T, dtype=dtype),
        "values": torch.zeros(B, T, dtype=dtype),
        "old_values": torch.zeros(B, T, dtype=dtype),
        "mask": torch.ones(B, T, dtype=torch.bool),
        "vocab_logits": torch.randn(B, T, V, dtype=dtype),
        **_DEFAULTS,
    }
    kwargs.update(overrides)
    return compute_ppo_loss(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("dtype", _DTYPES)
def test_uniform_policy_logprobs_zero(dtype: torch.dtype) -> None:
    if not _supported(dtype):
        pytest.skip("float16 unsupported on CPU")
    out = _loss(dtype, logprobs=torch.zeros(B, T, dtype=dtype))
    assert torch.isfinite(out.total_loss)


@pytest.mark.parametrize("dtype", _DTYPES)
def test_near_deterministic_extreme_logprobs(dtype: torch.dtype) -> None:
    if not _supported(dtype):
        pytest.skip("float16 unsupported on CPU")
    extreme = torch.full((B, T), -1e4, dtype=dtype)
    out = _loss(dtype, logprobs=extreme, old_logprobs=extreme.clone())
    assert torch.isfinite(out.total_loss)


@pytest.mark.parametrize("dtype", _DTYPES)
def test_zero_advantages_no_nan(dtype: torch.dtype) -> None:
    if not _supported(dtype):
        pytest.skip("float16 unsupported on CPU")
    lp = torch.zeros(B, T, dtype=dtype, requires_grad=True)
    out = compute_ppo_loss(
        logprobs=lp,
        old_logprobs=torch.zeros(B, T, dtype=dtype),
        advantages=torch.zeros(B, T, dtype=dtype),
        returns=torch.zeros(B, T, dtype=dtype),
        values=torch.zeros(B, T, dtype=dtype),
        old_values=torch.zeros(B, T, dtype=dtype),
        mask=torch.ones(B, T, dtype=torch.bool),
        vocab_logits=torch.randn(B, T, V, dtype=dtype),
        **_DEFAULTS,
    )
    out.total_loss.backward()
    assert torch.isfinite(out.total_loss)
    assert lp.grad is not None
    assert torch.isfinite(lp.grad).all()


def test_zero_variance_returns_explained_variance_zero() -> None:
    values = torch.randn(16)
    constant_returns = torch.full((16,), 3.0)
    ev = explained_variance(values, constant_returns)
    assert ev == 0.0


@pytest.mark.parametrize("dtype", _DTYPES)
def test_length_one_sequences(dtype: torch.dtype) -> None:
    if not _supported(dtype):
        pytest.skip("float16 unsupported on CPU")
    rewards = torch.ones(3, 1, dtype=dtype)
    values = torch.zeros(3, 1, dtype=dtype)
    mask = torch.ones(3, 1, dtype=torch.bool)
    adv, ret = compute_gae(rewards, values, mask, gamma=0.99, lam=0.95)
    assert torch.isfinite(adv).all()
    assert torch.isfinite(ret).all()
    out = _loss(
        dtype,
        logprobs=torch.zeros(3, 1, dtype=dtype),
        old_logprobs=torch.zeros(3, 1, dtype=dtype),
        advantages=adv,
        returns=ret,
        values=torch.zeros(3, 1, dtype=dtype),
        old_values=torch.zeros(3, 1, dtype=dtype),
        mask=mask,
        vocab_logits=torch.randn(3, 1, V, dtype=dtype),
    )
    assert torch.isfinite(out.total_loss)


@pytest.mark.parametrize("dtype", _DTYPES)
def test_all_padding_batch_zero_loss(dtype: torch.dtype) -> None:
    if not _supported(dtype):
        pytest.skip("float16 unsupported on CPU")
    out = _loss(dtype, mask=torch.zeros(B, T, dtype=torch.bool))
    assert torch.isfinite(out.total_loss)
    assert float(out.total_loss) == pytest.approx(0.0)
