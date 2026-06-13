"""Unit tests for rlhf.utils."""

from __future__ import annotations

import torch

from rlhf.utils import count_parameters, reproducibility_info, set_seed


def test_set_seed_makes_rng_deterministic() -> None:
    set_seed(123)
    a = torch.randn(5)
    set_seed(123)
    b = torch.randn(5)
    assert torch.equal(a, b)


def test_set_seed_deterministic_flag() -> None:
    set_seed(7, deterministic=True)
    assert torch.backends.cudnn.deterministic is True
    # Reset to avoid leaking the deterministic flag into other tests.
    torch.backends.cudnn.deterministic = False


def test_count_parameters() -> None:
    model = torch.nn.Linear(4, 3)  # 4*3 weights + 3 bias = 15
    counts = count_parameters(model)
    assert counts["total"] == 15
    assert counts["trainable"] == 15
    for p in model.parameters():
        p.requires_grad_(False)
    assert count_parameters(model)["trainable"] == 0


def test_reproducibility_info() -> None:
    info = reproducibility_info({"lr": 1e-4})
    assert "torch_version" in info
    assert "transformers_version" in info
    assert info["config"] == {"lr": 1e-4}
