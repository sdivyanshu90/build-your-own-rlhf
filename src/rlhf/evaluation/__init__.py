"""evaluation — metrics and evaluation loops for RLHF policies."""

from __future__ import annotations

from rlhf.evaluation.evaluator import EvalReport, Evaluator
from rlhf.evaluation.human_eval import HumanEvalSample, export_for_human_eval
from rlhf.evaluation.metrics import (
    approx_kl,
    clip_fraction,
    explained_variance,
    kl_divergence,
    perplexity,
    response_length_stats,
    reward_hacking_score,
    reward_win_rate,
)

__all__ = [
    "EvalReport",
    "Evaluator",
    "HumanEvalSample",
    "approx_kl",
    "clip_fraction",
    "explained_variance",
    "export_for_human_eval",
    "kl_divergence",
    "perplexity",
    "response_length_stats",
    "reward_hacking_score",
    "reward_win_rate",
]
