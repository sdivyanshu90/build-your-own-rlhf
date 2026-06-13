"""
evaluation.metrics — diagnostic metrics for PPO / RLHF training and evaluation.

Overview
--------
A collection of small, well-tested metric functions used both inside the PPO
loop (clip fraction, approximate KL, explained variance) and during offline
evaluation (win rate, perplexity, response-length statistics, reward-hacking
score). Every function is implemented in ``torch`` so it composes with autograd
where that is meaningful, and every function returns a finite value for empty /
fully-masked inputs rather than raising or producing NaN.

Mathematical Background
-----------------------
Exact categorical KL (full distributions, last dim = vocabulary)::

    KL(p || q) = sum_a p(a) * (log p(a) - log q(a))

Token-level KL estimate (per-action log-probs)::

    KL_hat = E_t[ log p(a_t) - log q(a_t) ]

Explained variance::

    EV = 1 - Var(returns - values) / Var(returns)

Usage Example
-------------
>>> import torch
>>> from rlhf.evaluation.metrics import explained_variance
>>> explained_variance(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([1.0, 2.0, 3.0]))
1.0

Legend: B = batch, T = sequence length, V = vocabulary size.
"""

from __future__ import annotations

import torch
from torch import Tensor

# Small constant guarding divisions whose denominator can legitimately be zero
# (e.g. zero-variance returns, zero mean reward).
_EPS: float = 1e-8


def _masked_mean(x: Tensor, mask: Tensor) -> Tensor:
    """Mean of ``x`` over True positions of ``mask``; 0.0 when the mask is empty."""
    mask_f = mask.to(x.dtype)
    return (x * mask_f).sum() / mask_f.sum().clamp(min=1.0)


def _variance(x: Tensor) -> Tensor:
    """Population variance of a flat tensor; 0.0 for fewer than two elements."""
    if x.numel() < 2:
        return torch.zeros((), dtype=x.dtype, device=x.device)
    return x.var(unbiased=False)


def kl_divergence(p_logprobs: Tensor, q_logprobs: Tensor, mask: Tensor) -> float:
    """
    KL divergence between policy ``p`` and reference ``q``.

    Dual-mode:

    * **3-D inputs** ``(..., V)`` are treated as full log-probability
      distributions; the exact categorical KL is summed over the vocabulary and
      averaged over masked positions.
    * **2-D inputs** ``(B, T)`` are treated as per-action log-probs; the
      token-level estimate ``mean(p - q)`` is returned.

    Args:
        p_logprobs: Policy log-probs, ``(..., V)`` or ``(B, T)``.
        q_logprobs: Reference log-probs, same shape as ``p_logprobs``.
        mask: Boolean mask over the leading (non-vocab) dimensions.

    Returns:
        Mean KL in nats as a Python float (0.0 for an empty mask).
    """
    if p_logprobs.shape != q_logprobs.shape:
        raise ValueError("p_logprobs and q_logprobs must share a shape.")
    if p_logprobs.dim() >= 3:
        # Exact KL per position: sum_a exp(p) * (p - q) over the vocab dim.
        per_pos = (p_logprobs.exp() * (p_logprobs - q_logprobs)).sum(dim=-1)
        return float(_masked_mean(per_pos, mask))
    # Token-level sample estimate of KL(p || q).
    return float(_masked_mean(p_logprobs - q_logprobs, mask))


def approx_kl(old_logprobs: Tensor, new_logprobs: Tensor, mask: Tensor) -> float:
    """
    Cheap PPO-internal KL estimate ``mean(old_logprobs - new_logprobs)``.

    Args:
        old_logprobs: ``(B, T)`` log-probs collected during rollout.
        new_logprobs: ``(B, T)`` current-policy log-probs.
        mask: ``(B, T)`` bool.

    Returns:
        Masked-mean KL estimate as a float (0.0 for an empty mask).
    """
    return float(_masked_mean(old_logprobs - new_logprobs, mask))


def explained_variance(values: Tensor, returns: Tensor) -> float:
    """
    Fraction of the return variance explained by the value function.

    Args:
        values: ``(N,)`` (or any shape) value predictions.
        returns: Same shape as ``values``; the regression targets.

    Returns:
        ``1 - Var(returns - values) / Var(returns)``; 0.0 when ``Var(returns)``
        is zero (an uninformative baseline), never NaN.
    """
    values = values.reshape(-1)
    returns = returns.reshape(-1)
    var_returns = _variance(returns)
    if float(var_returns) <= 0.0:
        return 0.0
    return float(1.0 - _variance(returns - values) / var_returns)


def clip_fraction(ratios: Tensor, eps: float) -> float:
    """
    Fraction of timesteps whose probability ratio left ``[1-eps, 1+eps]``.

    Args:
        ratios: Probability ratios ``r_t(theta)`` of any shape.
        eps: PPO clip range epsilon.

    Returns:
        Fraction in ``[0, 1]`` (0.0 for an empty tensor).
    """
    if ratios.numel() == 0:
        return 0.0
    clipped = (torch.abs(ratios - 1.0) > eps).to(torch.float32)
    return float(clipped.mean())


def reward_win_rate(rewards_a: Tensor, rewards_b: Tensor) -> float:
    """
    Probability that policy A out-rewards policy B, counting ties as 0.5.

    Args:
        rewards_a: ``(N,)`` rewards for policy A.
        rewards_b: ``(N,)`` rewards for policy B (paired with A).

    Returns:
        ``P(r_a > r_b) + 0.5 * P(r_a == r_b)`` as a float; 0.5 for empty input.
    """
    if rewards_a.numel() == 0:
        return 0.5
    if rewards_a.shape != rewards_b.shape:
        raise ValueError("reward_win_rate requires paired, equal-shaped inputs.")
    wins = (rewards_a > rewards_b).to(torch.float32)
    ties = (rewards_a == rewards_b).to(torch.float32)
    return float((wins + 0.5 * ties).mean())


def response_length_stats(lengths: Tensor) -> dict[str, float]:
    """
    Summary statistics of response lengths.

    Args:
        lengths: ``(N,)`` response token counts.

    Returns:
        Dict with keys ``mean, std, p50, p95, p99`` (all zeros for empty input).
    """
    if lengths.numel() == 0:
        return {"mean": 0.0, "std": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0}
    x = lengths.to(torch.float32)
    quantiles = torch.quantile(x, torch.tensor([0.50, 0.95, 0.99]))
    return {
        "mean": float(x.mean()),
        "std": float(x.std(unbiased=False)),
        "p50": float(quantiles[0]),
        "p95": float(quantiles[1]),
        "p99": float(quantiles[2]),
    }


def reward_hacking_score(rm_rewards: Tensor, rm_std: Tensor) -> float:
    """
    Proxy for reward over-optimization: mean ensemble std over mean reward.

    A large value means the reward ensemble disagrees strongly relative to the
    reward magnitude — a canary for the policy exploiting a single model.

    Args:
        rm_rewards: ``(N,)`` mean ensemble rewards.
        rm_std: ``(N,)`` ensemble standard deviations.

    Returns:
        ``mean(rm_std) / (|mean(rm_rewards)| + eps)`` as a float (0.0 if empty).
    """
    if rm_rewards.numel() == 0:
        return 0.0
    mean_reward = rm_rewards.to(torch.float32).mean().abs()
    mean_std = rm_std.to(torch.float32).mean()
    return float(mean_std / (mean_reward + _EPS))


def perplexity(logprobs: Tensor, lengths: Tensor | None = None) -> float:
    """
    Token-level perplexity ``exp(-mean log p)``.

    Args:
        logprobs: ``(N,)`` per-token log-probabilities.
        lengths: Optional ``(M,)`` per-sequence token counts; if given, the mean
            is taken over ``lengths.sum()`` tokens rather than ``logprobs.numel()``.

    Returns:
        Perplexity as a float; 1.0 for empty input.
    """
    if logprobs.numel() == 0:
        return 1.0
    total_logprob = logprobs.to(torch.float32).sum()
    if lengths is not None and lengths.numel() > 0:
        total_tokens = float(lengths.to(torch.float32).sum().clamp(min=1.0))
    else:
        total_tokens = float(logprobs.numel())
    return float(torch.exp(-total_logprob / total_tokens))


__all__ = [
    "approx_kl",
    "clip_fraction",
    "explained_variance",
    "kl_divergence",
    "perplexity",
    "response_length_stats",
    "reward_hacking_score",
    "reward_win_rate",
]
