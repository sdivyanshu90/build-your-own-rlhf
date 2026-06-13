"""
data.schemas — Pydantic v2 models for preference data and PPO rollouts.

Overview
--------
These models are the typed contract between every stage of the pipeline:

* :class:`Preference` — one human preference comparison ``(prompt, chosen, rejected)``
  used to train the Bradley-Terry reward model.
* :class:`Prompt` — a single inference prompt sampled during PPO rollout collection.
* :class:`Rollout` — one complete generated response with all per-token tensors
  (log-probs, reference log-probs, values) needed for advantage estimation.
* :class:`PPOBatch` — a collection of rollouts plus aggregate statistics.

All models forbid extra fields and validate field-level and cross-field invariants
(non-empty text, equal-length per-token arrays) at construction time, so malformed
data fails fast at the boundary rather than deep inside a training loop.

Usage Example
-------------
>>> from rlhf.data.schemas import Preference
>>> p = Preference(prompt="2+2?", chosen="4", rejected="5", annotator_id="a1")
>>> p.chosen
'4'

References
----------
- Bradley & Terry (1952). Rank analysis of incomplete block designs.
- Stiennon et al. (2020). Learning to summarize from human feedback.
  https://arxiv.org/abs/2009.01325
"""

from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Preference(BaseModel):
    """A single human preference comparison over two responses to one prompt."""

    model_config = ConfigDict(extra="forbid")

    id: UUID = Field(default_factory=uuid4, description="Auto-generated unique identifier.")
    prompt: str = Field(..., description="The shared prompt shown to the annotator.")
    chosen: str = Field(..., description="The response the annotator preferred (y_w).")
    rejected: str = Field(..., description="The response the annotator rejected (y_l).")
    annotator_id: str = Field(..., description="Stable identifier of the human annotator.")
    score_chosen: float | None = Field(
        None, description="Optional scalar quality score for the chosen response."
    )
    score_rejected: float | None = Field(
        None, description="Optional scalar quality score for the rejected response."
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Free-form provenance metadata."
    )

    @field_validator("chosen", "rejected", "prompt", "annotator_id")
    @classmethod
    def not_empty(cls, v: str) -> str:
        """Reject blank or whitespace-only text fields."""
        # A preference pair with an empty side carries no learnable signal and
        # would silently produce a zero-margin Bradley-Terry target, so we reject
        # it at the boundary instead of letting it pollute training.
        if not v or not v.strip():
            raise ValueError("text field must be a non-empty, non-whitespace string")
        return v


class Prompt(BaseModel):
    """A single prompt sampled during PPO rollout collection."""

    model_config = ConfigDict(extra="forbid")

    id: UUID = Field(default_factory=uuid4, description="Auto-generated unique identifier.")
    text: str = Field(..., description="The prompt text to condition generation on.")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Free-form metadata (source, difficulty, ...)."
    )

    @field_validator("text")
    @classmethod
    def not_empty(cls, v: str) -> str:
        """Reject blank or whitespace-only prompt text."""
        if not v or not v.strip():
            raise ValueError("prompt text must be a non-empty, non-whitespace string")
        return v


class Rollout(BaseModel):
    """
    One complete generated response with all tensors required for a PPO update.

    The per-token arrays ``response_ids``, ``logprobs``, ``ref_logprobs`` and
    ``values`` are aligned: index ``t`` describes the ``t``-th generated token.
    ``advantages`` and ``returns`` are filled in later by the rollout buffer and
    must, when present, match the response length.
    """

    model_config = ConfigDict(extra="forbid")

    prompt_ids: list[int] = Field(..., description="Token ids of the conditioning prompt.")
    response_ids: list[int] = Field(..., description="Token ids of the generated response.")
    logprobs: list[float] = Field(..., description="log pi_theta(a_t | s_t) per response token.")
    ref_logprobs: list[float] = Field(..., description="log pi_ref(a_t | s_t) per response token.")
    values: list[float] = Field(..., description="V(s_t) per response token.")
    reward: float = Field(..., description="Scalar reward-model score for the full response.")
    advantages: list[float] = Field(
        default_factory=list, description="GAE advantages (filled by the buffer)."
    )
    returns: list[float] = Field(
        default_factory=list, description="Discounted returns (filled by the buffer)."
    )

    @model_validator(mode="after")
    def _validate_aligned_lengths(self) -> Rollout:
        # response_ids, logprobs, ref_logprobs and values are produced one-per
        # generated token; a length mismatch means rollout collection corrupted
        # the alignment and every downstream advantage would be meaningless.
        n = len(self.response_ids)
        for name in ("logprobs", "ref_logprobs", "values"):
            arr = getattr(self, name)
            if len(arr) != n:
                raise ValueError(
                    f"'{name}' has length {len(arr)} but response_ids has length {n}; "
                    "all per-token arrays must be aligned."
                )
        # advantages / returns are optional but, once set, must also align.
        for name in ("advantages", "returns"):
            arr = getattr(self, name)
            if arr and len(arr) != n:
                raise ValueError(f"'{name}' has length {len(arr)} but response_ids has length {n}.")
        return self

    @property
    def response_length(self) -> int:
        """Number of generated response tokens (excludes the prompt)."""
        return len(self.response_ids)


class PPOBatch(BaseModel):
    """A batch of rollouts plus aggregate statistics for one PPO update phase."""

    model_config = ConfigDict(extra="forbid")

    rollouts: list[Rollout] = Field(
        default_factory=list, description="The rollouts collected this phase."
    )
    stats: dict[str, float] = Field(
        default_factory=dict, description="Aggregate scalar statistics for logging."
    )

    def __len__(self) -> int:
        return len(self.rollouts)


__all__ = ["PPOBatch", "Preference", "Prompt", "Rollout"]
