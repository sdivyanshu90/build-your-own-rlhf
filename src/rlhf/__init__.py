"""
rlhf — a production-grade Reinforcement Learning from Human Feedback pipeline.

Overview
--------
This package implements the full RLHF stack centred on a from-scratch Proximal
Policy Optimization (PPO) trainer:

* ``rlhf.config``      — Pydantic-validated hyperparameter schema.
* ``rlhf.data``        — preference / prompt datasets, collators, preprocessing.
* ``rlhf.models``      — policy (+ value head), reward model, reference model.
* ``rlhf.training``    — SFT, reward-model, and PPO trainers.
* ``rlhf.inference``   — batched autoregressive generation with log-prob capture.
* ``rlhf.evaluation``  — KL / win-rate / reward / perplexity metrics + evaluator.
* ``rlhf.security``    — checkpoint integrity and prompt-injection guards.
* ``rlhf.monitoring``  — dual W&B / TensorBoard logging and anomaly alerts.

Importing :mod:`rlhf` has **zero side effects** — no model downloads, no device
allocation, no global state mutation. Sub-modules are imported lazily by callers.

Usage Example
-------------
>>> import rlhf
>>> rlhf.__version__
'0.1.0'
"""

from __future__ import annotations

from rlhf.exceptions import RLHFError

__version__ = "0.1.0"

__all__ = ["RLHFError", "__version__"]
