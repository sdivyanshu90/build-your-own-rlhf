"""
utils — cross-cutting helpers (seeding, reproducibility, parameter counts).

Overview
--------
Small, dependency-light utilities used by every trainer and script:

* :func:`set_seed` — seed Python, NumPy and PyTorch (and optionally enforce
  deterministic cuDNN) so a run is reproducible.
* :func:`count_parameters` — count total / trainable parameters of a module.
* :func:`reproducibility_info` — capture library versions + config for run logs.
"""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np
import torch
from torch import nn


def set_seed(seed: int, deterministic: bool = False) -> None:
    """
    Seed all RNGs for reproducibility.

    Args:
        seed: The seed value.
        deterministic: If True, also set ``PYTHONHASHSEED`` and force
            deterministic cuDNN (slower but bit-reproducible on GPU).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ["PYTHONHASHSEED"] = str(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def count_parameters(module: nn.Module) -> dict[str, int]:
    """Return ``{"total": ..., "trainable": ...}`` parameter counts for ``module``."""
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def reproducibility_info(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Capture library versions and config for logging at the start of a run."""
    import transformers

    info: dict[str, Any] = {
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if config is not None:
        info["config"] = config
    return info


__all__ = ["count_parameters", "reproducibility_info", "set_seed"]
