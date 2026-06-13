"""
data.preprocessing — tokenization helpers shared across datasets.

Overview
--------
Pure functions that turn text into token-id tensors with the conventions the
pipeline expects:

* :func:`encode_text` — encode a single string to ids (with optional truncation).
* :func:`encode_pair` — encode ``prompt + response`` for reward / SFT scoring.
* :func:`build_generation_inputs` — **left**-pad a batch of prompts so that
  ``model.generate`` appends new tokens at a common offset.
* :func:`pad_sequences` — right-pad a list of variable-length id lists.

Keeping these as free functions (rather than tokenizer subclasses) makes them
trivially unit-testable and keeps datasets/collators thin.

Legend: B = batch, L = sequence length.
"""

from __future__ import annotations

import torch
from torch import Tensor
from transformers import PreTrainedTokenizerBase


def encode_text(
    tokenizer: PreTrainedTokenizerBase,
    text: str,
    max_length: int | None = None,
) -> list[int]:
    """Encode ``text`` to token ids, optionally truncating to ``max_length``."""
    ids: list[int] = tokenizer.encode(
        text,
        truncation=max_length is not None,
        max_length=max_length,
        add_special_tokens=False,
    )
    return ids


def encode_pair(
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    response: str,
    max_length: int,
    append_eos: bool = True,
) -> list[int]:
    """
    Encode a ``prompt`` + ``response`` into a single id sequence.

    An EOS token is appended (when available) so the reward model has a
    well-defined final position to pool, and SFT learns to terminate.
    """
    ids: list[int] = tokenizer.encode(prompt + response, truncation=True, max_length=max_length)
    eos = tokenizer.eos_token_id
    if append_eos and eos is not None and (not ids or ids[-1] != eos):
        ids = [*ids[: max_length - 1], eos]
    return ids


def pad_sequences(
    sequences: list[list[int]],
    pad_id: int,
    padding_side: str = "right",
) -> tuple[Tensor, Tensor]:
    """
    Pad variable-length id lists into ``(B, L)`` tensors.

    Args:
        sequences: List of token-id lists.
        pad_id: Padding token id.
        padding_side: ``"right"`` (default) or ``"left"``.

    Returns:
        ``(input_ids, attention_mask)`` both ``(B, L)`` long tensors.
    """
    if not sequences:
        return torch.empty(0, 0, dtype=torch.long), torch.empty(0, 0, dtype=torch.long)
    max_len = max(len(s) for s in sequences)
    max_len = max(max_len, 1)
    input_ids = torch.full((len(sequences), max_len), pad_id, dtype=torch.long)
    attention = torch.zeros(len(sequences), max_len, dtype=torch.long)
    for i, seq in enumerate(sequences):
        length = len(seq)
        if length == 0:
            continue
        if padding_side == "left":
            input_ids[i, max_len - length :] = torch.tensor(seq, dtype=torch.long)
            attention[i, max_len - length :] = 1
        else:
            input_ids[i, :length] = torch.tensor(seq, dtype=torch.long)
            attention[i, :length] = 1
    return input_ids, attention


def build_generation_inputs(
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
    max_length: int,
    device: torch.device | None = None,
) -> tuple[Tensor, Tensor]:
    """
    Left-pad a batch of prompts for batched autoregressive generation.

    Left padding is required so every prompt's final real token sits at the same
    column; otherwise ``generate`` would append continuations at ragged offsets.
    """
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    encoded = [encode_text(tokenizer, p, max_length=max_length) for p in prompts]
    input_ids, attention = pad_sequences(encoded, pad_id=pad_id, padding_side="left")
    if device is not None:
        input_ids = input_ids.to(device)
        attention = attention.to(device)
    return input_ids, attention


__all__ = [
    "build_generation_inputs",
    "encode_pair",
    "encode_text",
    "pad_sequences",
]
