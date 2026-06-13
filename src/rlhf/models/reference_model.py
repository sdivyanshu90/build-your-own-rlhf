"""
models.reference_model — the frozen reference policy pi_ref.

Overview
--------
PPO-for-RLHF penalizes divergence from a fixed reference policy (typically the
SFT model). :class:`ReferenceModel` wraps a frozen causal-LM backbone and exposes
:meth:`compute_logprobs`, returning per-token log-probabilities aligned exactly
the way :meth:`PolicyModel.score_sequence` aligns them, so the per-token KL
``log pi_theta - log pi_ref`` lines up token-for-token.

The reference is always in eval mode with ``requires_grad=False``; no gradients
ever flow through it. :meth:`from_policy` snapshots a policy's backbone (a deep
copy) so the reference is decoupled from subsequent policy updates.

Legend: B = batch, L = sequence length, V = vocabulary size.
"""

from __future__ import annotations

import copy

import torch
from torch import Tensor, nn
from transformers import PretrainedConfig, PreTrainedModel

from rlhf.exceptions import PolicyModelError
from rlhf.models.base import load_causal_lm, logprobs_from_logits, resolve_dtype


class ReferenceModel(nn.Module):
    """A frozen causal-LM used to compute reference log-probabilities."""

    def __init__(self, backbone: PreTrainedModel) -> None:
        super().__init__()
        self.backbone = backbone
        # The reference never trains: detach it from autograd and pin eval mode.
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad_(False)

    @classmethod
    def from_config(cls, config: PretrainedConfig) -> ReferenceModel:
        """Build a frozen reference with random weights from a config."""
        return cls(load_causal_lm(None, config=config))

    @classmethod
    def from_pretrained(
        cls, model_name_or_path: str, dtype: str | torch.dtype | None = None
    ) -> ReferenceModel:
        """Build a frozen reference from a pretrained path/name."""
        return cls(load_causal_lm(model_name_or_path, dtype=resolve_dtype(dtype)))

    @classmethod
    def from_policy(cls, policy: nn.Module) -> ReferenceModel:
        """
        Snapshot a policy's backbone into a frozen reference.

        A deep copy decouples the reference from later in-place policy updates.
        (In production one may instead share the frozen weights copy-on-write to
        halve memory; the deep copy is the safe, correctness-first default.)
        """
        backbone = getattr(policy, "backbone", None)
        if backbone is None:
            raise PolicyModelError("policy has no 'backbone' attribute to snapshot.")
        return cls(copy.deepcopy(backbone))

    @property
    def device(self) -> torch.device:
        """Device of the model parameters."""
        return next(self.parameters()).device

    @torch.no_grad()
    def compute_logprobs(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """
        Per-token reference log-probabilities, right-aligned to token positions.

        Args:
            input_ids: ``(B, L)`` full prompt+response token ids.
            attention_mask: ``(B, L)`` mask.

        Returns:
            ``(B, L)`` where index ``p`` is ``log pi_ref`` of the realized token
            at position ``p`` (position 0 is 0.0 — no token precedes it).
        """
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        logits = outputs.logits
        logprobs = torch.zeros_like(logits[:, :, 0])
        logprobs[:, 1:] = logprobs_from_logits(logits[:, :-1, :], input_ids[:, 1:])
        return logprobs


__all__ = ["ReferenceModel"]
