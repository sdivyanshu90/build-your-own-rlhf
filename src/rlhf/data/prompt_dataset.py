"""
data.prompt_dataset — torch Dataset over prompts for PPO rollout collection.

Overview
--------
:class:`PromptDataset` holds the prompts the policy generates from during PPO.
It accepts raw strings or :class:`~rlhf.data.schemas.Prompt` objects and yields
plain prompt text; tokenization / left-padding for generation is performed by the
:class:`~rlhf.data.collators.PromptCollator` (so the same dataset can feed models
with different tokenizers).
"""

from __future__ import annotations

from torch.utils.data import Dataset

from rlhf.data.schemas import Prompt
from rlhf.exceptions import DataValidationError


class PromptDataset(Dataset[str]):
    """A map-style dataset of prompt strings."""

    def __init__(self, prompts: list[str] | list[Prompt]) -> None:
        if not prompts:
            raise DataValidationError("PromptDataset requires at least one prompt.")
        # Normalize to strings up front so __getitem__ is trivial and typed.
        self.prompts: list[str] = [p.text if isinstance(p, Prompt) else p for p in prompts]

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, index: int) -> str:
        return self.prompts[index]


__all__ = ["PromptDataset"]
