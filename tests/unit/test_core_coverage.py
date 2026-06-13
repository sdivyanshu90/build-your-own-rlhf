"""Edge-case tests bringing the PPO business-logic modules to 100% coverage."""

from __future__ import annotations

import pytest
import torch

from rlhf.evaluation import metrics
from rlhf.exceptions import KLControllerError, RolloutError
from rlhf.training.ppo.kl_controller import AdaptiveKLController, FixedKLController
from rlhf.training.ppo.rollout import RolloutBuffer, compute_gae, whiten


# ----------------------------- metrics edge cases -----------------------------
def test_kl_divergence_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="share a shape"):
        metrics.kl_divergence(
            torch.randn(2, 3), torch.randn(2, 4), torch.ones(2, 3, dtype=torch.bool)
        )


def test_reward_win_rate_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="paired"):
        metrics.reward_win_rate(torch.randn(5), torch.randn(4))


# -------------------------- KL controller edge cases --------------------------
def test_fixed_controller_rejects_nan_update() -> None:
    with pytest.raises(KLControllerError):
        FixedKLController(coef=0.2).update(float("nan"))


def test_fixed_controller_state_round_trip() -> None:
    ctrl = FixedKLController(coef=0.3)
    restored = FixedKLController(coef=0.9)
    restored.load_state_dict(ctrl.state_dict())
    assert restored.value == pytest.approx(0.3)


def test_adaptive_rejects_init_outside_bounds() -> None:
    with pytest.raises(KLControllerError, match="must lie within"):
        AdaptiveKLController(init_coef=1.0, target=6.0, step_size=0.1, coef_min=0.05, coef_max=0.5)


# ----------------------------- rollout edge cases -----------------------------
def test_compute_gae_shape_mismatch_raises() -> None:
    with pytest.raises(RolloutError, match="shape mismatch"):
        compute_gae(
            torch.randn(2, 3), torch.randn(2, 4), torch.ones(2, 3, dtype=torch.bool), 1.0, 1.0
        )


def test_whiten_keeps_mean_when_not_shifting() -> None:
    values = torch.tensor([[1.0, 3.0, 5.0]])
    mask = torch.ones(1, 3, dtype=torch.bool)
    shifted = whiten(values, mask, shift_mean=True)
    unshifted = whiten(values, mask, shift_mean=False)
    # Not shifting the mean leaves a non-zero mean; shifting drives it to ~0.
    assert float(shifted.mean()) == pytest.approx(0.0, abs=1e-5)
    assert float(unshifted.mean()) == pytest.approx(float(values.mean()), abs=1e-5)


def test_buffer_rejects_nonpositive_capacity() -> None:
    with pytest.raises(RolloutError, match="capacity"):
        RolloutBuffer(capacity=0)


def test_get_mini_batches_rejects_nonpositive_size(filled_buffer: RolloutBuffer) -> None:
    filled_buffer.compute_advantages(gamma=0.99, lam=0.95, kl_coef=0.2)
    with pytest.raises(RolloutError, match="mini_batch_size"):
        next(filled_buffer.get_mini_batches(mini_batch_size=0))


def test_compute_advantages_without_whitening(filled_buffer: RolloutBuffer) -> None:
    # With whitening disabled the advantage mean is generally non-zero (it is not
    # forced to zero-mean/unit-variance), exercising the no-whiten branch.
    stats = filled_buffer.compute_advantages(
        gamma=0.99, lam=0.95, kl_coef=0.2, whiten_advantages=False
    )
    assert "advantage_mean" in stats
    assert "advantage_std" in stats


def test_buffer_handles_zero_length_response() -> None:
    buffer = RolloutBuffer(capacity=2)
    buffer.push(
        prompt_ids=torch.tensor([1, 2]),
        response_ids=torch.tensor([3, 4]),
        logprobs=-torch.rand(2),
        ref_logprobs=-torch.rand(2),
        values=torch.randn(2),
        reward=1.0,
    )
    # A degenerate rollout with an empty response must not crash advantage/GAE.
    buffer.push(
        prompt_ids=torch.tensor([1, 2]),
        response_ids=torch.tensor([], dtype=torch.long),
        logprobs=torch.tensor([]),
        ref_logprobs=torch.tensor([]),
        values=torch.tensor([]),
        reward=0.5,
    )
    buffer.compute_advantages(gamma=0.99, lam=0.95, kl_coef=0.2)
    batches = list(buffer.get_mini_batches(mini_batch_size=2, shuffle=False))
    assert len(batches) == 1
    # The empty-response row contributes no response-mask positions.
    assert batches[0].response_mask.sum() >= 1
