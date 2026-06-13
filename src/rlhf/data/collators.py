"""
data.collators — batch collation for preference and prompt datasets.

Overview
--------
* :class:`PreferenceCollator` pads the ``chosen`` and ``rejected`` id lists of a
  batch into aligned ``(B, L)`` tensors for Bradley-Terry training.
* :class:`PromptCollator` tokenizes and **left**-pads a batch of prompt strings
  for batched generation.

Legend: B = batch, L = sequence length.
"""

from __future__ import annotations

from torch import Tensor
from transformers import PreTrainedTokenizerBase

from rlhf.data.preprocessing import encode_text, pad_sequences


def _resolve_pad_id(tokenizer: PreTrainedTokenizerBase) -> int:
    """Pad id, falling back to EOS then 0 when no pad token is configured."""
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    return int(pad_id)


class PreferenceCollator:
    """Collate tokenized preference pairs into padded tensor batches."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self.pad_id = _resolve_pad_id(tokenizer)

    def __call__(self, batch: list[dict[str, list[int]]]) -> dict[str, Tensor]:
        chosen = [item["chosen_input_ids"] for item in batch]
        rejected = [item["rejected_input_ids"] for item in batch]
        chosen_ids, chosen_mask = pad_sequences(chosen, self.pad_id, padding_side="right")
        rejected_ids, rejected_mask = pad_sequences(rejected, self.pad_id, padding_side="right")
        return {
            "chosen_input_ids": chosen_ids,
            "chosen_attention_mask": chosen_mask,
            "rejected_input_ids": rejected_ids,
            "rejected_attention_mask": rejected_mask,
        }


class PromptCollator:
    """Tokenize and left-pad a batch of prompt strings for generation."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase, max_length: int = 256) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_id = _resolve_pad_id(tokenizer)

    def __call__(self, prompts: list[str]) -> dict[str, Tensor]:
        encoded = [encode_text(self.tokenizer, p, max_length=self.max_length) for p in prompts]
        input_ids, attention = pad_sequences(encoded, self.pad_id, padding_side="left")
        return {"input_ids": input_ids, "attention_mask": attention}


def collate_response_ids(response_ids: list[list[int]], pad_id: int) -> tuple[Tensor, Tensor]:
    """Right-pad a batch of response id lists (used in tests / scoring)."""
    return pad_sequences(response_ids, pad_id, padding_side="right")


__all__ = ["PreferenceCollator", "PromptCollator", "collate_response_ids"]
