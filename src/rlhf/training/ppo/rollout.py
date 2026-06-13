"""
training.ppo.rollout — rollout storage, RLHF reward shaping, and GAE.

Overview
--------
This module owns everything that happens *between* generating responses and
running the PPO gradient update:

1. :class:`RolloutBuffer` — a fixed-capacity store of complete response rollouts.
2. :func:`compute_rlhf_rewards` — turns the scalar reward-model score plus the
   per-token KL penalty into a dense per-token reward signal.
3. :func:`compute_gae` — Generalized Advantage Estimation, fully vectorized.

Mathematical Background
-----------------------
Per-token RLHF reward (only the final token carries the RM score)::

    r_t = r_RM * 1[t == T] - beta * KL_t,   KL_t = log pi_theta - log pi_ref

GAE (Schulman et al. 2016)::

    delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
    A_t     = sum_{l>=0} (gamma*lambda)^l * delta_{t+l}

We evaluate the closed form ``A = delta @ W`` with a discount matrix
``W[i, j] = (gamma*lambda)^(j-i)`` for ``j >= i`` (else 0). This is loop-free,
numerically stable for all ``gamma*lambda in [0, 1]`` (every weight lies in
``[0, 1]``), and — unlike the reversed-cumsum/division trick — remains exact at
``gamma*lambda = 0`` (the matrix degenerates to the identity).

Usage Example
-------------
>>> import torch
>>> from rlhf.training.ppo.rollout import compute_gae
>>> rewards = torch.tensor([[1.0, 1.0, 1.0]])
>>> values = torch.zeros(1, 3)
>>> mask = torch.ones(1, 3, dtype=torch.bool)
>>> adv, ret = compute_gae(rewards, values, mask, gamma=1.0, lam=1.0)
>>> ret.tolist()
[[3.0, 2.0, 1.0]]

References
----------
- Schulman et al. (2016). High-Dimensional Continuous Control Using GAE.
  https://arxiv.org/abs/1506.02438

Legend: B = batch, T = response length, P = prompt length, L = full seq length.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import torch
from torch import Tensor

from rlhf.exceptions import BufferFullError, RolloutError

# Numerical floor added to the variance during advantage whitening so that a
# zero-variance advantage batch normalizes to zeros instead of dividing by zero.
_WHITEN_EPS: float = 1e-8


@dataclass
class RolloutMiniBatch:
    """
    One mini-batch of rollouts, expressed over full (prompt+response) sequences.

    All tensors share the leading shape ``(b, L)`` where ``b`` is the mini-batch
    size and ``L`` the padded full-sequence length. ``response_mask`` marks the
    generated tokens (the PPO actions); every per-token quantity is non-zero only
    at those positions.

    Attributes:
        input_ids: ``(b, L)`` long — prompt tokens followed by response tokens.
        attention_mask: ``(b, L)`` bool — True for real tokens, False for padding.
        response_mask: ``(b, L)`` bool — True only at generated response tokens.
        old_logprobs: ``(b, L)`` — log pi_theta_old at response positions.
        ref_logprobs: ``(b, L)`` — log pi_ref at response positions.
        old_values: ``(b, L)`` — V_old at response positions.
        advantages: ``(b, L)`` — GAE advantages at response positions.
        returns: ``(b, L)`` — discounted returns at response positions.
    """

    input_ids: Tensor
    attention_mask: Tensor
    response_mask: Tensor
    old_logprobs: Tensor
    ref_logprobs: Tensor
    old_values: Tensor
    advantages: Tensor
    returns: Tensor

    def to(self, device: torch.device) -> RolloutMiniBatch:
        """Return a copy with every tensor moved to ``device``."""
        return RolloutMiniBatch(
            input_ids=self.input_ids.to(device),
            attention_mask=self.attention_mask.to(device),
            response_mask=self.response_mask.to(device),
            old_logprobs=self.old_logprobs.to(device),
            ref_logprobs=self.ref_logprobs.to(device),
            old_values=self.old_values.to(device),
            advantages=self.advantages.to(device),
            returns=self.returns.to(device),
        )


def compute_gae(
    rewards: Tensor,
    values: Tensor,
    mask: Tensor,
    gamma: float,
    lam: float,
) -> tuple[Tensor, Tensor]:
    """
    Vectorized Generalized Advantage Estimation over a padded batch.

    Args:
        rewards: ``(B, T)`` per-token rewards (KL penalty already folded in).
        values: ``(B, T)`` value estimates ``V(s_t)``.
        mask: ``(B, T)`` bool; True for real tokens. Padding must be right-aligned.
        gamma: Discount factor in ``[0, 1]``.
        lam: GAE smoothing parameter in ``[0, 1]``.

    Returns:
        ``(advantages, returns)``, each ``(B, T)``. Padded positions are zero.

    Notes:
        The terminal token bootstraps from ``V = 0`` (episode ends at EOS); this
        falls out automatically because right-padded value positions are zeroed
        before the one-step shift.
    """
    if rewards.shape != values.shape or rewards.shape != mask.shape:
        raise RolloutError(
            f"compute_gae shape mismatch: rewards {tuple(rewards.shape)}, "
            f"values {tuple(values.shape)}, mask {tuple(mask.shape)}."
        )
    mask_f = mask.to(rewards.dtype)
    t = rewards.shape[1]

    # Zero out padded values so the one-step look-ahead V(s_{t+1}) never reads a
    # garbage/padding value and the last real token bootstraps from 0.
    values_masked = values * mask_f
    # next_values[:, t] = V(s_{t+1}); shift left by one, pad the tail with 0.
    next_values = torch.cat([values_masked[:, 1:], torch.zeros_like(values_masked[:, :1])], dim=1)

    # TD residual delta_t = r_t + gamma * V(s_{t+1}) - V(s_t), masked so padded
    # positions contribute exactly zero to every downstream advantage.
    deltas = (rewards + gamma * next_values - values_masked) * mask_f

    # Discount matrix W[i, j] = (gamma*lambda)^(j-i) for j >= i, else 0. Building
    # the closed-form weighting as a matmul keeps everything vectorized (no loop
    # over time) and every entry stays within [0, 1] so there is no overflow —
    # even when gamma*lambda == 0, where W collapses to the identity.
    gl = gamma * lam
    idx = torch.arange(t, device=rewards.device)
    exponent = idx.unsqueeze(0) - idx.unsqueeze(1)  # exponent[i, j] = j - i
    upper = exponent >= 0
    weight = torch.zeros((t, t), dtype=rewards.dtype, device=rewards.device)
    weight[upper] = gl ** exponent[upper].to(rewards.dtype)

    # advantages[b, i] = sum_j deltas[b, j] * W[i, j]  ==  deltas @ W^T
    advantages = (deltas @ weight.t()) * mask_f
    # returns = advantage + value baseline (the value target for L_VF).
    returns = (advantages + values_masked) * mask_f
    return advantages, returns


def compute_rlhf_rewards(
    scalar_rewards: Tensor,
    logprobs: Tensor,
    ref_logprobs: Tensor,
    mask: Tensor,
    kl_coef: float,
) -> tuple[Tensor, Tensor]:
    """
    Build the dense per-token RLHF reward and the per-token KL penalty.

    Args:
        scalar_rewards: ``(B,)`` reward-model score for each full response.
        logprobs: ``(B, T)`` log pi_theta at each response token.
        ref_logprobs: ``(B, T)`` log pi_ref at each response token.
        mask: ``(B, T)`` bool; True for real tokens (right-padded).
        kl_coef: Current KL penalty coefficient ``beta``.

    Returns:
        ``(rewards, kl)`` each ``(B, T)``. ``rewards`` carries ``-beta*KL`` at
        every token plus the scalar RM score at the final real token of each row.
    """
    mask_f = mask.to(logprobs.dtype)
    # Per-token KL, restricted to real tokens — padding KL is meaningless and
    # would otherwise leak a penalty into masked positions.
    kl = (logprobs - ref_logprobs) * mask_f
    rewards = -kl_coef * kl

    # Add the scalar RM reward at the LAST real token of each sequence (the EOS
    # position). lengths-1 gives that index; clamp guards all-padding rows.
    lengths = mask_f.sum(dim=1).long()
    last_idx = (lengths - 1).clamp(min=0)
    rows = torch.arange(scalar_rewards.shape[0], device=scalar_rewards.device)
    has_tokens = lengths > 0
    # Only write the terminal reward where the row actually has tokens.
    rewards[rows[has_tokens], last_idx[has_tokens]] += scalar_rewards[has_tokens].to(rewards.dtype)
    return rewards, kl


def whiten(values: Tensor, mask: Tensor, shift_mean: bool = True) -> Tensor:
    """
    Normalize ``values`` to zero mean / unit variance over masked positions.

    Args:
        values: ``(B, T)`` tensor to normalize.
        mask: ``(B, T)`` bool; only True positions contribute to the statistics.
        shift_mean: If False, divide by std but keep the original mean.

    Returns:
        Whitened ``(B, T)`` tensor with padded positions re-zeroed.
    """
    mask_f = mask.to(values.dtype)
    n = mask_f.sum().clamp(min=1.0)
    mean = (values * mask_f).sum() / n
    var = (((values - mean) * mask_f) ** 2).sum() / n
    whitened = (values - mean) * torch.rsqrt(var + _WHITEN_EPS)
    if not shift_mean:
        whitened = whitened + mean
    return whitened * mask_f


class RolloutBuffer:
    """
    Fixed-capacity store of complete response rollouts for one PPO phase.

    The buffer holds raw CPU tensors during collection, then — in
    :meth:`compute_advantages` — pads them into batched ``(N, T)`` tensors,
    applies the RLHF reward shaping and GAE, and whitens the advantages. After
    that, :meth:`get_mini_batches` yields :class:`RolloutMiniBatch` objects over
    full prompt+response sequences for the inner PPO epochs.

    Args:
        capacity: Maximum number of rollouts the buffer can hold.
        pad_token_id: Token id used to right-pad sequences.

    Raises:
        BufferFullError: When :meth:`push` is called on a full buffer.
        RolloutError: When advantages are requested before any rollout is stored,
            or mini-batches are requested before advantages are computed.
    """

    def __init__(self, capacity: int, pad_token_id: int = 0) -> None:
        if capacity <= 0:
            raise RolloutError(f"capacity must be positive, got {capacity}.")
        self.capacity = capacity
        self.pad_token_id = pad_token_id
        self._prompt_ids: list[Tensor] = []
        self._response_ids: list[Tensor] = []
        self._logprobs: list[Tensor] = []
        self._ref_logprobs: list[Tensor] = []
        self._values: list[Tensor] = []
        self._rewards: list[float] = []
        # Filled by compute_advantages(); response-space (N, T) tensors.
        self._padded: dict[str, Tensor] | None = None

    def __len__(self) -> int:
        return len(self._response_ids)

    @property
    def is_full(self) -> bool:
        """True when the buffer holds ``capacity`` rollouts."""
        return len(self) >= self.capacity

    def push(
        self,
        prompt_ids: Tensor,
        response_ids: Tensor,
        logprobs: Tensor,
        ref_logprobs: Tensor,
        values: Tensor,
        reward: float,
    ) -> None:
        """
        Store one rollout. All per-token tensors must share the response length.

        Args:
            prompt_ids: ``(P,)`` long — conditioning prompt token ids.
            response_ids: ``(T,)`` long — generated response token ids.
            logprobs: ``(T,)`` — log pi_theta at each response token.
            ref_logprobs: ``(T,)`` — log pi_ref at each response token.
            values: ``(T,)`` — value estimate at each response token.
            reward: Scalar reward-model score for the full response.
        """
        if self.is_full:
            raise BufferFullError(
                f"buffer holds {len(self)}/{self.capacity} rollouts; cannot push more."
            )
        n = response_ids.shape[0]
        for name, arr in (
            ("logprobs", logprobs),
            ("ref_logprobs", ref_logprobs),
            ("values", values),
        ):
            if arr.shape[0] != n:
                raise RolloutError(f"'{name}' length {arr.shape[0]} != response length {n}.")
        # Detach + move to CPU so the buffer never pins activations on the GPU.
        self._prompt_ids.append(prompt_ids.detach().cpu().long())
        self._response_ids.append(response_ids.detach().cpu().long())
        self._logprobs.append(logprobs.detach().cpu().float())
        self._ref_logprobs.append(ref_logprobs.detach().cpu().float())
        self._values.append(values.detach().cpu().float())
        self._rewards.append(float(reward))
        # Any state computed for a previous phase is now stale.
        self._padded = None

    def _pad_response_space(self) -> dict[str, Tensor]:
        """Pad stored per-token tensors into ``(N, T)`` response-space batches."""
        n = len(self)
        if n == 0:
            raise RolloutError("cannot compute advantages on an empty buffer.")
        t_max = max(int(r.shape[0]) for r in self._response_ids)
        t_max = max(t_max, 1)  # keep a non-degenerate time dimension
        logprobs = torch.zeros(n, t_max)
        ref_logprobs = torch.zeros(n, t_max)
        values = torch.zeros(n, t_max)
        mask = torch.zeros(n, t_max, dtype=torch.bool)
        for i in range(n):
            length = int(self._response_ids[i].shape[0])
            if length == 0:
                continue
            logprobs[i, :length] = self._logprobs[i]
            ref_logprobs[i, :length] = self._ref_logprobs[i]
            values[i, :length] = self._values[i]
            mask[i, :length] = True
        return {
            "logprobs": logprobs,
            "ref_logprobs": ref_logprobs,
            "values": values,
            "mask": mask,
            "scalar_rewards": torch.tensor(self._rewards, dtype=torch.float32),
        }

    def compute_advantages(
        self, gamma: float, lam: float, kl_coef: float, whiten_advantages: bool = True
    ) -> dict[str, float]:
        """
        Apply RLHF reward shaping, GAE, and advantage whitening to the buffer.

        Args:
            gamma: GAE discount factor.
            lam: GAE smoothing parameter.
            kl_coef: Current KL penalty coefficient ``beta``.
            whiten_advantages: Whether to zero-mean/unit-variance the advantages.

        Returns:
            A dict of summary statistics (mean reward, mean KL, advantage mean/std).
        """
        padded = self._pad_response_space()
        rewards, kl = compute_rlhf_rewards(
            padded["scalar_rewards"],
            padded["logprobs"],
            padded["ref_logprobs"],
            padded["mask"],
            kl_coef,
        )
        advantages, returns = compute_gae(rewards, padded["values"], padded["mask"], gamma, lam)
        mask = padded["mask"]
        if whiten_advantages:
            advantages = whiten(advantages, mask, shift_mean=True)

        padded["advantages"] = advantages
        padded["returns"] = returns
        padded["kl"] = kl
        padded["rewards"] = rewards
        self._padded = padded

        mask_f = mask.to(torch.float32)
        denom = mask_f.sum().clamp(min=1.0)
        return {
            "reward_mean": float(padded["scalar_rewards"].mean()),
            "reward_std": float(padded["scalar_rewards"].std(unbiased=False)),
            "kl_mean": float(kl.sum() / denom),
            "advantage_mean": float((advantages * mask_f).sum() / denom),
            "advantage_std": float(
                (((advantages - (advantages * mask_f).sum() / denom) * mask_f) ** 2).sum() / denom
            )
            ** 0.5,
        }

    def get_mini_batches(
        self,
        mini_batch_size: int,
        shuffle: bool = True,
        device: torch.device | None = None,
        generator: torch.Generator | None = None,
    ) -> Iterator[RolloutMiniBatch]:
        """
        Yield mini-batches over full prompt+response sequences.

        Each rollout appears exactly once. The final mini-batch may be smaller
        than ``mini_batch_size`` if ``len(buffer)`` is not divisible by it.

        Args:
            mini_batch_size: Number of rollouts per mini-batch.
            shuffle: Shuffle rollout order before batching.
            device: Optional device to move each mini-batch onto.
            generator: Optional RNG for deterministic shuffling.

        Yields:
            :class:`RolloutMiniBatch` instances.
        """
        if self._padded is None:
            raise RolloutError("call compute_advantages() before get_mini_batches().")
        if mini_batch_size <= 0:
            raise RolloutError(f"mini_batch_size must be positive, got {mini_batch_size}.")
        n = len(self)
        order = torch.randperm(n, generator=generator) if shuffle else torch.arange(n)
        for start in range(0, n, mini_batch_size):
            idx = order[start : start + mini_batch_size].tolist()
            yield self._assemble_mini_batch(idx, device)

    def _assemble_mini_batch(
        self, indices: list[int], device: torch.device | None
    ) -> RolloutMiniBatch:
        """Concatenate prompt+response for ``indices`` and pad to a common length."""
        assert self._padded is not None  # guarded by caller
        full_seqs: list[Tensor] = []
        resp_lengths: list[int] = []
        prompt_lengths: list[int] = []
        for i in indices:
            prompt = self._prompt_ids[i]
            response = self._response_ids[i]
            full_seqs.append(torch.cat([prompt, response], dim=0))
            prompt_lengths.append(int(prompt.shape[0]))
            resp_lengths.append(int(response.shape[0]))

        b = len(indices)
        l_max = max(int(s.shape[0]) for s in full_seqs)
        l_max = max(l_max, 1)
        input_ids = torch.full((b, l_max), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros(b, l_max, dtype=torch.bool)
        response_mask = torch.zeros(b, l_max, dtype=torch.bool)
        old_logprobs = torch.zeros(b, l_max)
        ref_logprobs = torch.zeros(b, l_max)
        old_values = torch.zeros(b, l_max)
        advantages = torch.zeros(b, l_max)
        returns = torch.zeros(b, l_max)

        for row, i in enumerate(indices):
            seq = full_seqs[row]
            seq_len = int(seq.shape[0])
            p_len = prompt_lengths[row]
            r_len = resp_lengths[row]
            input_ids[row, :seq_len] = seq
            attention_mask[row, :seq_len] = True
            if r_len == 0:
                continue
            # Response tokens occupy full-sequence positions [p_len, p_len+r_len).
            response_mask[row, p_len : p_len + r_len] = True
            old_logprobs[row, p_len : p_len + r_len] = self._padded["logprobs"][i, :r_len]
            ref_logprobs[row, p_len : p_len + r_len] = self._padded["ref_logprobs"][i, :r_len]
            old_values[row, p_len : p_len + r_len] = self._padded["values"][i, :r_len]
            advantages[row, p_len : p_len + r_len] = self._padded["advantages"][i, :r_len]
            returns[row, p_len : p_len + r_len] = self._padded["returns"][i, :r_len]

        batch = RolloutMiniBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            response_mask=response_mask,
            old_logprobs=old_logprobs,
            ref_logprobs=ref_logprobs,
            old_values=old_values,
            advantages=advantages,
            returns=returns,
        )
        return batch.to(device) if device is not None else batch

    def clear(self) -> None:
        """Reset the buffer to empty so it can be reused for the next phase."""
        self._prompt_ids.clear()
        self._response_ids.clear()
        self._logprobs.clear()
        self._ref_logprobs.clear()
        self._values.clear()
        self._rewards.clear()
        self._padded = None


__all__ = [
    "RolloutBuffer",
    "RolloutMiniBatch",
    "compute_gae",
    "compute_rlhf_rewards",
    "whiten",
]
