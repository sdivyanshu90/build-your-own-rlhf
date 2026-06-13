"""
models.base — shared model interface and tensor utilities.

Overview
--------
Defines the :class:`RLHFModel` structural protocol implemented by every model
wrapper, plus a handful of small, well-tested tensor helpers used across the
policy, reward, and reference models:

* :func:`logprobs_from_logits` — gather per-token log-probabilities of the taken
  actions from a logits tensor.
* :func:`last_token_indices` — locate the final non-padding position of each row
  (the EOS position used for reward pooling).
* :func:`load_causal_lm` / :func:`load_base_model` — construct a HuggingFace
  backbone from a path/name **or** a config (the latter avoids any download and
  is what the tests use).

Legend: B = batch, T = sequence length, V = vocabulary size, H = hidden size.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    PretrainedConfig,
    PreTrainedModel,
)

_DTYPE_MAP: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def resolve_dtype(dtype: str | torch.dtype | None) -> torch.dtype | None:
    """Map a dtype string (``"bfloat16"``) to a ``torch.dtype`` (``None`` passes through)."""
    if dtype is None or isinstance(dtype, torch.dtype):
        return dtype
    if dtype not in _DTYPE_MAP:
        raise ValueError(f"Unsupported dtype '{dtype}'. Choose from {sorted(_DTYPE_MAP)}.")
    return _DTYPE_MAP[dtype]


@runtime_checkable
class RLHFModel(Protocol):
    """Structural interface common to every RLHF model wrapper."""

    backbone: nn.Module

    @property
    def device(self) -> torch.device:
        """Device the model parameters live on."""
        ...

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None) -> object:
        """Run a forward pass over a batch of token ids."""
        ...


def logprobs_from_logits(logits: Tensor, labels: Tensor) -> Tensor:
    """
    Per-token log-probability of ``labels`` under ``logits``.

    Args:
        logits: ``(B, T, V)`` unnormalized logits.
        labels: ``(B, T)`` token ids whose log-prob to extract.

    Returns:
        ``(B, T)`` log-probabilities ``log softmax(logits)[b, t, labels[b, t]]``.
    """
    log_probs = F.log_softmax(logits, dim=-1)
    return torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)


def last_token_indices(attention_mask: Tensor) -> Tensor:
    """
    Index of the last real (non-padding) token in each row.

    Args:
        attention_mask: ``(B, T)`` mask, 1/True for real tokens. Padding may be
            on either side; this counts real tokens and assumes left-to-right
            packing of the real tokens (standard right-padding).

    Returns:
        ``(B,)`` long indices, clamped to ``>= 0`` for all-padding rows.
    """
    lengths = attention_mask.to(torch.long).sum(dim=1)
    return (lengths - 1).clamp(min=0)


def load_causal_lm(
    model_name_or_path: str | None,
    config: PretrainedConfig | None = None,
    dtype: torch.dtype | None = None,
    revision: str | None = None,
) -> PreTrainedModel:
    """
    Build a causal-LM backbone from a pretrained path/name or a config.

    Exactly one of ``model_name_or_path`` / ``config`` must be provided. The
    config path instantiates **random** weights (no download) and is what the
    test-suite uses for its tiny GPT-2. Pin ``revision`` (a commit SHA / tag) in
    production to make Hub downloads reproducible and tamper-evident.
    """
    if (model_name_or_path is None) == (config is None):
        raise ValueError("Provide exactly one of model_name_or_path or config.")
    if config is not None:
        return AutoModelForCausalLM.from_config(config)
    kwargs = {"torch_dtype": dtype} if dtype is not None else {}
    # nosec B615: revision pinning is exposed to the caller (config-driven); we
    # do not hardcode a revision because the backbone is user-selected.
    return AutoModelForCausalLM.from_pretrained(  # nosec B615
        model_name_or_path, revision=revision, **kwargs
    )


def load_base_model(
    model_name_or_path: str | None,
    config: PretrainedConfig | None = None,
    dtype: torch.dtype | None = None,
    revision: str | None = None,
) -> PreTrainedModel:
    """Build a hidden-states-only backbone (no LM head) from a path or config."""
    if (model_name_or_path is None) == (config is None):
        raise ValueError("Provide exactly one of model_name_or_path or config.")
    if config is not None:
        return AutoModel.from_config(config)
    kwargs = {"torch_dtype": dtype} if dtype is not None else {}
    # nosec B615: see load_causal_lm — revision pinning is the caller's choice.
    return AutoModel.from_pretrained(model_name_or_path, revision=revision, **kwargs)  # nosec B615


__all__ = [
    "RLHFModel",
    "last_token_indices",
    "load_base_model",
    "load_causal_lm",
    "logprobs_from_logits",
    "resolve_dtype",
]
