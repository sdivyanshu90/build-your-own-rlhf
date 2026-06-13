"""
models.reward_model — scalar reward head, Bradley-Terry loss, and ensemble.

Overview
--------
:class:`RewardModel` places a scalar reward head on top of a (optionally frozen)
transformer backbone and pools the representation at the **last non-padding
token** — the only position where a reward is well-defined for variable-length
responses (Stiennon et al. 2020). It maintains online (Welford) reward statistics
so rewards can be standardized before they are fed to the PPO policy.

:class:`RewardModelEnsemble` aggregates ``N`` independently-initialized reward
models and returns ``(mean, std)``; the spread is an uncertainty signal and a
reward-hacking canary.

Mathematical Background
-----------------------
Bradley-Terry preference loss::

    L_BT = -E_{(x, y_w, y_l)}[ log sigmoid( r(x, y_w) - r(x, y_l) ) ]

implemented with the numerically stable ``F.logsigmoid``.

Welford online mean/variance (Chan's parallel form) lets us fold a whole batch
into the running statistics in one shot without storing samples.

References
----------
- Bradley & Terry (1952). Rank analysis of incomplete block designs.
- Stiennon et al. (2020). https://arxiv.org/abs/2009.01325

Legend: B = batch, L = sequence length, H = hidden size, N = ensemble size.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import PretrainedConfig, PreTrainedModel

from rlhf.exceptions import RewardModelError
from rlhf.models.base import last_token_indices, load_base_model, resolve_dtype

# Floor on the running std during normalization to avoid divide-by-zero before
# enough reward samples have accumulated.
_NORM_EPS: float = 1e-6


def bradley_terry_loss(chosen_rewards: Tensor, rejected_rewards: Tensor) -> Tensor:
    """
    Mean Bradley-Terry loss over a batch of preference pairs.

    Args:
        chosen_rewards: ``(B,)`` rewards for the preferred responses.
        rejected_rewards: ``(B,)`` rewards for the rejected responses.

    Returns:
        Scalar loss ``-mean(log sigmoid(r_chosen - r_rejected))``. Equals
        ``log 2`` when the two rewards are equal everywhere.
    """
    if chosen_rewards.shape != rejected_rewards.shape:
        raise RewardModelError("chosen and rejected rewards must share a shape.")
    return -F.logsigmoid(chosen_rewards - rejected_rewards).mean()


class RunningMoments(nn.Module):
    """Online mean / variance tracker (batched Welford) stored as buffers."""

    # Class-level annotations so type-checkers know the buffer element types.
    count: Tensor
    mean: Tensor
    m2: Tensor

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("count", torch.zeros((), dtype=torch.float64))
        self.register_buffer("mean", torch.zeros((), dtype=torch.float64))
        self.register_buffer("m2", torch.zeros((), dtype=torch.float64))

    @torch.no_grad()
    def update(self, x: Tensor) -> None:
        """Fold a batch ``x`` into the running statistics."""
        flat = x.detach().reshape(-1).to(torch.float64)
        n_b = flat.numel()
        if n_b == 0:
            return
        mean_b = flat.mean()
        m2_b = flat.var(unbiased=False) * n_b
        # Chan's parallel combination of two sample sets.
        delta = mean_b - self.mean
        total = self.count + n_b
        self.mean = self.mean + delta * n_b / total
        self.m2 = self.m2 + m2_b + delta**2 * self.count * n_b / total
        self.count = total

    @property
    def variance(self) -> Tensor:
        """Population variance of the samples seen so far (0 until 1 sample)."""
        if float(self.count) < 1:
            return torch.zeros((), dtype=torch.float64)
        var: Tensor = self.m2 / self.count
        return var

    @property
    def std(self) -> Tensor:
        """Standard deviation of the samples seen so far."""
        return torch.sqrt(self.variance)

    def normalize(self, x: Tensor) -> Tensor:
        """Standardize ``x`` with the running mean/std (identity before any data)."""
        if float(self.count) < 1:
            return x
        normalized: Tensor = (x - self.mean.to(x.dtype)) / (self.std.to(x.dtype) + _NORM_EPS)
        return normalized


class RewardModel(nn.Module):
    """
    A scalar reward head on a transformer backbone with last-token pooling.

    Args:
        model_name_or_path: HF id/path (mutually exclusive with ``backbone``).
        backbone: Pre-built base model (used by :meth:`from_config`).
        freeze_backbone: Freeze backbone weights and train only the head.
        normalize_rewards: Apply running normalization inside :meth:`forward`.
        dtype: Optional weight dtype.
    """

    def __init__(
        self,
        model_name_or_path: str | None = None,
        *,
        backbone: PreTrainedModel | None = None,
        freeze_backbone: bool = True,
        normalize_rewards: bool = False,
        dtype: str | torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if backbone is None and model_name_or_path is None:
            raise RewardModelError("Provide either model_name_or_path or backbone.")
        self.backbone = (
            backbone
            if backbone is not None
            else load_base_model(model_name_or_path, dtype=resolve_dtype(dtype))
        )
        hidden_size = int(self.backbone.config.hidden_size)
        self.reward_pre_norm = nn.LayerNorm(hidden_size)
        self.reward_head = nn.Linear(hidden_size, 1)
        nn.init.normal_(self.reward_head.weight, std=1.0 / (hidden_size + 1) ** 0.5)
        nn.init.zeros_(self.reward_head.bias)
        self.normalizer = RunningMoments()
        self.normalize_rewards = normalize_rewards
        self._frozen = freeze_backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad_(False)
        if dtype is not None:
            resolved = resolve_dtype(dtype)
            self.reward_pre_norm = self.reward_pre_norm.to(resolved)
            self.reward_head = self.reward_head.to(resolved)

    @classmethod
    def from_config(
        cls,
        config: PretrainedConfig,
        freeze_backbone: bool = True,
        normalize_rewards: bool = False,
    ) -> RewardModel:
        """Build a reward model with random weights from a config (no download)."""
        backbone = load_base_model(None, config=config)
        return cls(
            backbone=backbone,
            freeze_backbone=freeze_backbone,
            normalize_rewards=normalize_rewards,
        )

    @property
    def device(self) -> torch.device:
        """Device of the model parameters."""
        return next(self.parameters()).device

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """
        Score a batch of sequences.

        Args:
            input_ids: ``(B, L)`` token ids.
            attention_mask: ``(B, L)`` mask.

        Returns:
            ``(B,)`` scalar rewards (normalized iff ``self.normalize_rewards``).
        """
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        hidden = outputs.last_hidden_state  # (B, L, H)
        # Pool the representation at the final real token (EOS position): the only
        # position that has attended to the entire variable-length response.
        idx = last_token_indices(attention_mask)  # (B,)
        rows = torch.arange(hidden.shape[0], device=hidden.device)
        pooled = hidden[rows, idx]  # (B, H)
        reward: Tensor = self.reward_head(self.reward_pre_norm(pooled)).squeeze(-1)  # (B,)
        if self.normalize_rewards:
            reward = self.normalizer.normalize(reward)
        return reward

    @torch.no_grad()
    def update_normalizer(self, rewards: Tensor) -> None:
        """Fold a batch of (raw) rewards into the running normalization stats."""
        self.normalizer.update(rewards)


class RewardModelEnsemble(nn.Module):
    """An ensemble of reward models returning ``(mean_reward, std_reward)``."""

    def __init__(self, models: list[RewardModel]) -> None:
        super().__init__()
        if len(models) < 1:
            raise RewardModelError("ensemble requires at least one reward model.")
        self.models = nn.ModuleList(models)

    @classmethod
    def from_config(
        cls,
        config: PretrainedConfig,
        ensemble_size: int,
        freeze_backbone: bool = True,
    ) -> RewardModelEnsemble:
        """Build ``ensemble_size`` independently-initialized reward models."""
        members: list[RewardModel] = []
        for i in range(ensemble_size):
            # Vary the seed per member so heads (and backbone, if unfrozen) differ.
            torch.manual_seed(1000 + i)
            members.append(RewardModel.from_config(config, freeze_backbone=freeze_backbone))
        return cls(members)

    @property
    def size(self) -> int:
        """Number of ensemble members."""
        return len(self.models)

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> tuple[Tensor, Tensor]:
        """
        Score a batch with every member.

        Returns:
            ``(mean, std)`` each ``(B,)``; ``std`` is zero for a single member.
        """
        rewards = torch.stack(
            [model(input_ids, attention_mask) for model in self.models], dim=0
        )  # (N, B)
        mean_reward: Tensor = rewards.mean(dim=0)
        std_reward: Tensor = rewards.std(dim=0, unbiased=False)
        return mean_reward, std_reward


__all__ = ["RewardModel", "RewardModelEnsemble", "RunningMoments", "bradley_terry_loss"]
