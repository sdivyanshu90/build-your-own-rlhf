"""Unit tests for data.preprocessing tokenization helpers."""

from __future__ import annotations

import torch
from transformers import PreTrainedTokenizerBase

from rlhf.data.preprocessing import (
    build_generation_inputs,
    encode_pair,
    encode_text,
    pad_sequences,
)


def test_pad_sequences_right() -> None:
    ids, mask = pad_sequences([[1, 2, 3], [4, 5]], pad_id=0, padding_side="right")
    assert ids.tolist() == [[1, 2, 3], [4, 5, 0]]
    assert mask.tolist() == [[1, 1, 1], [1, 1, 0]]


def test_pad_sequences_left() -> None:
    ids, mask = pad_sequences([[1, 2, 3], [4, 5]], pad_id=0, padding_side="left")
    assert ids.tolist() == [[1, 2, 3], [0, 4, 5]]
    assert mask.tolist() == [[1, 1, 1], [0, 1, 1]]


def test_pad_sequences_empty() -> None:
    ids, mask = pad_sequences([], pad_id=0)
    assert ids.numel() == 0
    assert mask.numel() == 0


def test_pad_sequences_handles_empty_row() -> None:
    ids, mask = pad_sequences([[], [1, 2]], pad_id=7, padding_side="right")
    assert ids.tolist() == [[7, 7], [1, 2]]
    assert mask.tolist() == [[0, 0], [1, 1]]


def test_encode_text_truncates(tokenizer: PreTrainedTokenizerBase) -> None:
    long_text = "word " * 100
    ids = encode_text(tokenizer, long_text, max_length=10)
    assert len(ids) <= 10


def test_encode_pair_appends_eos(tokenizer: PreTrainedTokenizerBase) -> None:
    ids = encode_pair(tokenizer, "Question? ", "Answer.", max_length=64)
    assert ids[-1] == tokenizer.eos_token_id


def test_build_generation_inputs_left_pads(tokenizer: PreTrainedTokenizerBase) -> None:
    prompts = ["short", "a much longer prompt with many tokens in it"]
    input_ids, attention = build_generation_inputs(tokenizer, prompts, max_length=32)
    assert input_ids.shape[0] == 2
    # Left padding: the shorter prompt has leading padding (mask starts with 0).
    assert attention[0, 0] == 0 or attention[0].sum() == attention[1].sum()
    # Every row's real tokens are right-aligned (last column is always attended).
    assert torch.all(attention[:, -1] == 1)
