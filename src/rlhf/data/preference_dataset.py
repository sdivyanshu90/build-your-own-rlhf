"""
data.preference_dataset — torch Dataset over human preference comparisons.

Overview
--------
:class:`PreferenceDataset` wraps a list of :class:`~rlhf.data.schemas.Preference`
records and a tokenizer, yielding, for each example, the tokenized
``prompt + chosen`` and ``prompt + rejected`` sequences consumed by the
Bradley-Terry reward-model trainer.

Usage Example
-------------
>>> # ds = PreferenceDataset(preferences, tokenizer, max_length=512)
>>> # item = ds[0]; item["chosen_input_ids"], item["rejected_input_ids"]
"""

from __future__ import annotations

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from rlhf.data.preprocessing import encode_pair
from rlhf.data.schemas import Preference
from rlhf.exceptions import DataValidationError


class PreferenceDataset(Dataset[dict[str, list[int]]]):
    """A map-style dataset of tokenized preference pairs."""

    def __init__(
        self,
        preferences: list[Preference],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 512,
    ) -> None:
        if not preferences:
            raise DataValidationError("PreferenceDataset requires at least one preference.")
        self.preferences = preferences
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.preferences)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        pref = self.preferences[index]
        chosen_ids = encode_pair(self.tokenizer, pref.prompt, pref.chosen, self.max_length)
        rejected_ids = encode_pair(self.tokenizer, pref.prompt, pref.rejected, self.max_length)
        return {"chosen_input_ids": chosen_ids, "rejected_input_ids": rejected_ids}


__all__ = ["PreferenceDataset"]
