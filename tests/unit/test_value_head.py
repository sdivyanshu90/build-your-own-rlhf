"""Unit tests for the PPO value head."""

from __future__ import annotations

import pytest
import torch

from rlhf.models.value_head import ValueHead


def test_output_shape() -> None:
    head = ValueHead(hidden_size=64, dropout=0.0)
    hidden = torch.randn(3, 7, 64)
    values = head(hidden)
    assert values.shape == (3, 7)


def test_zero_init_produces_zero_values() -> None:
    # The final linear is zero-initialized, so V(s) == 0 everywhere at step 0,
    # independent of the input hidden states.
    head = ValueHead(hidden_size=32, dropout=0.0)
    values = head(torch.randn(4, 6, 32))
    assert torch.all(values == 0.0)


def test_gradient_flows() -> None:
    head = ValueHead(hidden_size=16, dropout=0.0)
    hidden = torch.randn(2, 3, 16, requires_grad=True)
    head(hidden).sum().backward()
    assert hidden.grad is not None
    assert head.fc1.weight.grad is not None
    # Even with a zero-valued output, gradients still reach the output layer.
    assert head.fc2.weight.grad is not None


def test_odd_hidden_size_supported() -> None:
    head = ValueHead(hidden_size=15, dropout=0.0)
    assert head.fc1.out_features == 7  # floor(15 / 2)
    assert head(torch.randn(2, 4, 15)).shape == (2, 4)


def test_rejects_tiny_hidden_size() -> None:
    with pytest.raises(ValueError, match="hidden_size"):
        ValueHead(hidden_size=1)
