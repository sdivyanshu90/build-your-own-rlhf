"""
inference.stopping_criteria — custom generation stopping rules.

Overview
--------
Provides :class:`StopOnSequences`, a HuggingFace ``StoppingCriteria`` that halts
generation once every row in the batch has emitted one of a set of stop token
sequences (e.g. an end-of-turn marker). It complements the built-in
``max_new_tokens`` / EOS handling for chat-style or multi-token stop strings.

Legend: B = batch, L = sequence length.
"""

from __future__ import annotations

import torch
from torch import Tensor
from transformers import StoppingCriteria


class StopOnSequences(StoppingCriteria):  # type: ignore[misc]
    """
    Stop generation when each row ends with one of the given token sequences.

    Args:
        stop_sequences: List of token-id sequences; emitting any one ends a row.
        prompt_length: Number of prompt tokens (stop checks only the generated tail).
    """

    def __init__(self, stop_sequences: list[list[int]], prompt_length: int) -> None:
        super().__init__()
        if not stop_sequences or any(len(s) == 0 for s in stop_sequences):
            raise ValueError("stop_sequences must be a non-empty list of non-empty sequences.")
        self.stop_sequences = stop_sequences
        self.prompt_length = prompt_length

    def __call__(self, input_ids: Tensor, scores: Tensor, **kwargs: object) -> bool:
        generated = input_ids[:, self.prompt_length :]
        if generated.shape[1] == 0:
            return False
        # A row is "done" once its generated tail ends with any stop sequence.
        done = torch.zeros(generated.shape[0], dtype=torch.bool, device=generated.device)
        for stop in self.stop_sequences:
            length = len(stop)
            if generated.shape[1] < length:
                continue
            tail = generated[:, -length:]
            stop_tensor = torch.tensor(stop, device=generated.device, dtype=tail.dtype)
            done |= (tail == stop_tensor).all(dim=1)
        return bool(done.all())


__all__ = ["StopOnSequences"]
