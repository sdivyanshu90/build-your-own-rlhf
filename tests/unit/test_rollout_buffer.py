"""Unit tests for RolloutBuffer and RLHF reward shaping."""

from __future__ import annotations

import pytest
import torch

from rlhf.exceptions import BufferFullError, RolloutError
from rlhf.training.ppo.rollout import RolloutBuffer, compute_rlhf_rewards


def _push_one(buffer: RolloutBuffer, length: int = 4, reward: float = 1.0) -> None:
    buffer.push(
        prompt_ids=torch.tensor([1, 2, 3]),
        response_ids=torch.randint(3, 50, (length,)),
        logprobs=-torch.rand(length),
        ref_logprobs=-torch.rand(length),
        values=torch.randn(length),
        reward=reward,
    )


def test_buffer_raises_when_full() -> None:
    buffer = RolloutBuffer(capacity=2)
    _push_one(buffer)
    _push_one(buffer)
    assert buffer.is_full
    with pytest.raises(BufferFullError):
        _push_one(buffer)


def test_mini_batches_cover_all_rollouts_once(filled_buffer: RolloutBuffer) -> None:
    filled_buffer.compute_advantages(gamma=0.99, lam=0.95, kl_coef=0.2)
    total_rows = 0
    for batch in filled_buffer.get_mini_batches(mini_batch_size=3, shuffle=True):
        total_rows += batch.input_ids.shape[0]
        # Every mini-batch row must contain at least one response token.
        assert batch.response_mask.any(dim=1).all()
    assert total_rows == len(filled_buffer)


def test_advantage_normalization_zero_mean_unit_variance(filled_buffer: RolloutBuffer) -> None:
    stats = filled_buffer.compute_advantages(
        gamma=0.99, lam=0.95, kl_coef=0.2, whiten_advantages=True
    )
    assert stats["advantage_mean"] == pytest.approx(0.0, abs=1e-5)
    assert stats["advantage_std"] == pytest.approx(1.0, abs=1e-4)


def test_clear_resets_state(filled_buffer: RolloutBuffer) -> None:
    filled_buffer.compute_advantages(gamma=0.99, lam=0.95, kl_coef=0.2)
    filled_buffer.clear()
    assert len(filled_buffer) == 0
    assert not filled_buffer.is_full
    with pytest.raises(RolloutError):
        filled_buffer.compute_advantages(gamma=0.99, lam=0.95, kl_coef=0.2)


def test_kl_penalty_only_on_non_padding_positions() -> None:
    scalar_rewards = torch.tensor([2.0, 3.0])
    logprobs = torch.tensor([[-0.5, -0.5, -0.5], [-1.0, -1.0, -1.0]])
    ref_logprobs = torch.tensor([[-0.1, -0.1, -0.1], [-0.2, -0.2, -0.2]])
    mask = torch.tensor([[True, True, False], [True, False, False]])
    rewards, kl = compute_rlhf_rewards(scalar_rewards, logprobs, ref_logprobs, mask, kl_coef=0.5)
    # KL must be exactly zero at padded positions.
    assert kl[0, 2] == 0.0
    assert kl[1, 1] == 0.0
    assert kl[1, 2] == 0.0
    # Scalar reward lands on the LAST real token of each row.
    assert rewards[0, 1] == pytest.approx(-0.5 * (-0.5 - -0.1) + 2.0)
    assert rewards[1, 0] == pytest.approx(-0.5 * (-1.0 - -0.2) + 3.0)
    # Padded reward positions carry no signal.
    assert rewards[0, 2] == 0.0


def test_get_mini_batches_before_advantages_raises(filled_buffer: RolloutBuffer) -> None:
    with pytest.raises(RolloutError):
        next(filled_buffer.get_mini_batches(mini_batch_size=2))


def test_push_length_mismatch_raises() -> None:
    buffer = RolloutBuffer(capacity=2)
    with pytest.raises(RolloutError):
        buffer.push(
            prompt_ids=torch.tensor([1, 2]),
            response_ids=torch.tensor([3, 4, 5]),
            logprobs=-torch.rand(2),  # wrong length
            ref_logprobs=-torch.rand(3),
            values=torch.randn(3),
            reward=1.0,
        )
