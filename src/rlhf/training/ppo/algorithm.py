"""
training.ppo.algorithm — the core PPO clipped-surrogate loss.

Overview
--------
Implements the full PPO objective used for RLHF policy optimization:

    L_PPO = L_CLIP + c1 * L_VF - c2 * H

This module is deliberately model-agnostic: it operates purely on tensors of
log-probabilities, advantages, returns, values and vocabulary logits, so it can
be unit-tested without instantiating any language model.

Mathematical Background
-----------------------
Clipped policy surrogate (Schulman et al. 2017, eq. 7)::

    r_t(theta) = pi_theta(a_t | s_t) / pi_theta_old(a_t | s_t)
    L_CLIP    = E_t[ min( r_t * A_t, clip(r_t, 1-eps, 1+eps) * A_t ) ]

Clipped value loss::

    L_VF = E_t[ max( (V_theta - R_t)^2, (clip(V_theta, V_old +/- eps_v) - R_t)^2 ) ]

Entropy bonus over the vocabulary::

    H = E_t[ -sum_a pi_theta(a | s_t) log pi_theta(a | s_t) ]

Because optimizers *minimize*, the returned ``total_loss`` is
``policy_loss + c1 * value_loss - c2 * entropy`` where ``policy_loss = -L_CLIP``.

Usage Example
-------------
>>> import torch
>>> from rlhf.training.ppo.algorithm import compute_ppo_loss
>>> B, T, V = 2, 4, 7
>>> kw = dict(clip_eps=0.2, clip_eps_vf=0.2, entropy_coeff=0.01, value_coeff=0.5)
>>> out = compute_ppo_loss(
...     logprobs=torch.zeros(B, T), old_logprobs=torch.zeros(B, T),
...     advantages=torch.ones(B, T), returns=torch.zeros(B, T),
...     values=torch.zeros(B, T), old_values=torch.zeros(B, T),
...     mask=torch.ones(B, T, dtype=torch.bool),
...     vocab_logits=torch.zeros(B, T, V), **kw)
>>> bool(out.total_loss.isfinite())
True

References
----------
- Schulman et al. (2017). Proximal Policy Optimization Algorithms.
  https://arxiv.org/abs/1707.06347
- Ouyang et al. (2022). Training language models to follow instructions with
  human feedback. https://arxiv.org/abs/2203.02155

Legend: B = batch, T = sequence length, V = vocabulary size.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

# Maximum magnitude of the policy log-ratio before exponentiation. Under normal
# PPO dynamics |log r_t| is tiny (old ~ new), so this clamp is inert; it exists
# only to keep adversarial / near-deterministic inputs (e.g. logprobs = -1e4)
# from producing exp(+inf) and poisoning the loss with NaN.
_LOG_RATIO_CLAMP: float = 20.0


@dataclass(frozen=True)
class PPOLossOutput:
    """
    Result bundle from :func:`compute_ppo_loss`.

    Attributes:
        total_loss: Differentiable scalar tensor to call ``.backward()`` on.
        policy_loss: ``-L_CLIP`` (float, detached, for logging).
        value_loss: ``L_VF`` (float, detached).
        entropy_loss: Mean policy entropy ``H`` (float, positive; enters the
            total as ``-entropy_coeff * entropy_loss``).
        clip_fraction: Fraction of unmasked timesteps where the ratio was clipped.
        approx_kl: Cheap KL estimate ``mean(old_logprobs - logprobs)``.
        explained_variance: ``1 - Var(returns - values) / Var(returns)``.
    """

    total_loss: Tensor
    policy_loss: float
    value_loss: float
    entropy_loss: float
    clip_fraction: float
    approx_kl: float
    explained_variance: float


def _masked_mean(x: Tensor, mask: Tensor) -> Tensor:
    """
    Mean of ``x`` over positions where ``mask`` is True.

    Args:
        x: Tensor of shape ``(B, T)``.
        mask: Boolean / float tensor of shape ``(B, T)``; True = keep.

    Returns:
        Scalar tensor. Returns ``0.0`` (not NaN) when the mask is empty so that
        all-padding batches propagate a finite, differentiable zero.
    """
    mask_f = mask.to(x.dtype)
    denom = mask_f.sum()
    # Guard the division: an all-padding batch has denom == 0, which would yield
    # NaN. We keep the numerator in the graph (it is exactly zero) and divide by
    # a clamped denominator so gradients remain well-defined.
    return (x * mask_f).sum() / denom.clamp(min=1.0)


def _masked_variance(x: Tensor, mask: Tensor) -> Tensor:
    """Population variance of ``x`` over masked positions (0.0 if <2 elements)."""
    mask_f = mask.to(x.dtype)
    n = mask_f.sum()
    if n < 2:
        return torch.zeros((), dtype=x.dtype, device=x.device)
    mean = (x * mask_f).sum() / n
    centered = (x - mean) * mask_f
    variance: Tensor = (centered**2).sum() / n
    return variance


def _explained_variance(values: Tensor, returns: Tensor, mask: Tensor) -> Tensor:
    """
    ``1 - Var(returns - values) / Var(returns)`` over masked positions.

    Returns 0.0 (the uninformative-baseline value) when ``Var(returns)`` is zero,
    rather than dividing by zero.
    """
    var_returns = _masked_variance(returns, mask)
    if var_returns <= 0:
        return torch.zeros((), dtype=values.dtype, device=values.device)
    var_resid = _masked_variance(returns - values, mask)
    ev: Tensor = 1.0 - var_resid / var_returns
    return ev


def compute_ppo_loss(
    logprobs: Tensor,
    old_logprobs: Tensor,
    advantages: Tensor,
    returns: Tensor,
    values: Tensor,
    old_values: Tensor,
    mask: Tensor,
    clip_eps: float,
    clip_eps_vf: float,
    entropy_coeff: float,
    value_coeff: float,
    vocab_logits: Tensor,
) -> PPOLossOutput:
    """
    Compute the full clipped PPO loss with value and entropy terms.

    Args:
        logprobs: ``(B, T)`` current-policy log-probs of the taken actions.
        old_logprobs: ``(B, T)`` log-probs collected during rollout (detached).
        advantages: ``(B, T)`` GAE advantages (typically whitened).
        returns: ``(B, T)`` discounted returns (advantage + value target).
        values: ``(B, T)`` current value-head predictions.
        old_values: ``(B, T)`` value predictions collected during rollout.
        mask: ``(B, T)`` bool; True for real (non-padding) response tokens.
        clip_eps: Policy-ratio clip range epsilon.
        clip_eps_vf: Value-prediction clip range.
        entropy_coeff: Coefficient c2 on the entropy bonus.
        value_coeff: Coefficient c1 on the value loss.
        vocab_logits: ``(B, T, V)`` raw token logits, for the entropy term.

    Returns:
        :class:`PPOLossOutput` whose ``total_loss`` is differentiable.

    Notes:
        All reductions are masked so padding never contributes. An all-padding
        batch yields ``total_loss == 0`` rather than NaN.
    """
    # ----- Policy surrogate (L_CLIP) -------------------------------------------------
    # r_t(theta) = exp(log pi_theta - log pi_theta_old). The log-ratio is clamped
    # purely for numerical safety (see _LOG_RATIO_CLAMP); under normal training it
    # is a no-op because old and current policies are close.
    log_ratio = (logprobs - old_logprobs).clamp(-_LOG_RATIO_CLAMP, _LOG_RATIO_CLAMP)
    ratio = torch.exp(log_ratio)

    # We minimize, so the surrogate loss is -L_CLIP = max(-r*A, -clip(r)*A).
    # Taking max of the two negated terms is identical to negating the min of the
    # two positive terms in the PPO objective.
    pg_loss_unclipped = -advantages * ratio
    pg_loss_clipped = -advantages * torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    policy_loss = _masked_mean(torch.maximum(pg_loss_unclipped, pg_loss_clipped), mask)

    # ----- Clipped value loss (L_VF) -------------------------------------------------
    # The value prediction is clipped to a trust region around the rollout-time
    # value so a single noisy target cannot move V too far in one step. We take
    # the elementwise max of clipped/unclipped squared errors (pessimistic).
    values_clipped = old_values + torch.clamp(values - old_values, -clip_eps_vf, clip_eps_vf)
    vf_loss_unclipped = (values - returns) ** 2
    vf_loss_clipped = (values_clipped - returns) ** 2
    value_loss = _masked_mean(torch.maximum(vf_loss_unclipped, vf_loss_clipped), mask)

    # ----- Entropy bonus (H) ---------------------------------------------------------
    # Entropy is computed over the full vocabulary distribution at each position:
    # H_t = -sum_a softmax(logits)_a * log_softmax(logits)_a. log_softmax is used
    # for numerical stability (no explicit normalization / log of small numbers).
    token_log_probs = torch.log_softmax(vocab_logits, dim=-1)
    token_probs = token_log_probs.exp()
    per_token_entropy = -(token_probs * token_log_probs).sum(dim=-1)
    entropy = _masked_mean(per_token_entropy, mask)

    # ----- Total loss ----------------------------------------------------------------
    # L_PPO = L_CLIP + c1 * L_VF - c2 * H, with L_CLIP folded into policy_loss as
    # its negation. Higher entropy therefore lowers the total loss (exploration).
    total_loss = policy_loss + value_coeff * value_loss - entropy_coeff * entropy

    # ----- Diagnostics (no gradient) -------------------------------------------------
    with torch.no_grad():
        # approx_kl: cheap, biased estimate of KL(old || new) used to monitor the
        # update magnitude inside the inner PPO epochs (Schulman, blog 2020).
        approx_kl = _masked_mean(old_logprobs - logprobs, mask)
        # clip_fraction: how often the ratio left the trust region [1-eps, 1+eps].
        clipped = (torch.abs(ratio - 1.0) > clip_eps).to(ratio.dtype)
        clip_fraction = _masked_mean(clipped, mask)
        explained_var = _explained_variance(values, returns, mask)

    return PPOLossOutput(
        total_loss=total_loss,
        policy_loss=float(policy_loss.detach()),
        value_loss=float(value_loss.detach()),
        entropy_loss=float(entropy.detach()),
        clip_fraction=float(clip_fraction),
        approx_kl=float(approx_kl),
        explained_variance=float(explained_var),
    )


__all__ = ["PPOLossOutput", "compute_ppo_loss"]
