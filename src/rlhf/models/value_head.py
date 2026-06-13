"""
models.value_head — the PPO value head.

Overview
--------
A small MLP that maps a transformer's last-layer hidden states to a scalar value
estimate ``V(s_t)`` at every position. The architecture follows the spec:

    hidden_size -> hidden_size/2 -> 1

with a LayerNorm before each linear, a GELU non-linearity and dropout in the
middle, and a **zero-initialized** final linear so that values start at ~0. The
zero init matters: a value head that emits large random values early on injects
large, noisy advantages that destabilize the very first policy-gradient steps.

Legend: B = batch, T = sequence length, H = hidden size.
"""

from __future__ import annotations

from torch import Tensor, nn


class ValueHead(nn.Module):
    """
    Two-layer MLP value head with pre-LayerNorm and a zero-init output layer.

    Args:
        hidden_size: Width ``H`` of the incoming hidden states.
        dropout: Dropout probability applied after the first non-linearity.

    Forward:
        hidden_states: ``(B, T, H)`` last-layer hidden states.
        returns: ``(B, T)`` scalar value estimate per position.
    """

    def __init__(self, hidden_size: int, dropout: float = 0.1) -> None:
        super().__init__()
        if hidden_size < 2:
            raise ValueError(f"hidden_size must be >= 2, got {hidden_size}.")
        inner = max(hidden_size // 2, 1)
        self.pre_norm = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, inner)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.mid_norm = nn.LayerNorm(inner)
        self.fc2 = nn.Linear(inner, 1)
        self._init_weights()

    def _init_weights(self) -> None:
        # Standard small-gain init for the first projection; ZERO init for the
        # output so V(s) == 0 everywhere at step 0 and the advantage signal is
        # driven purely by the (also-zero) bootstrap until the head learns.
        nn.init.normal_(self.fc1.weight, std=1.0 / (self.fc1.in_features**0.5))
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Map ``(B, T, H)`` hidden states to ``(B, T)`` value estimates."""
        x = self.pre_norm(hidden_states)
        x = self.act(self.fc1(x))
        x = self.dropout(x)
        x = self.mid_norm(x)
        values: Tensor = self.fc2(x).squeeze(-1)
        return values


__all__ = ["ValueHead"]
