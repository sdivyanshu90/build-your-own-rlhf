#!/usr/bin/env python3
"""
evaluate.py — evaluate a trained policy against the reward model and reference.

Loads the policy, reference, reward model and a JSON-lines eval prompt set, runs
the :class:`~rlhf.evaluation.evaluator.Evaluator`, and prints / saves the report.

Usage:
    python scripts/evaluate.py --config config.yaml --prompts eval.jsonl
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
import typer
from transformers import AutoTokenizer

from rlhf.config.schema import RLHFConfig
from rlhf.evaluation.evaluator import Evaluator
from rlhf.models.policy import PolicyModel
from rlhf.models.reference_model import ReferenceModel
from rlhf.models.reward_model import RewardModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("rlhf.scripts.evaluate")


def _load_prompts(path: Path) -> list[str]:
    prompts: list[str] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                obj = json.loads(line)
                prompts.append(obj.get("prompt") or obj.get("text") or "")
    return [p for p in prompts if p]


def main(config: str, prompts: str = "eval.jsonl") -> None:
    """Evaluate the trained policy and write a JSON report."""
    cfg = RLHFConfig.from_yaml(config)
    device = torch.device(cfg.device)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    policy = PolicyModel(model_name_or_path=cfg.model.model_name_or_path)
    reference = ReferenceModel.from_policy(policy)
    reward_model = RewardModel(
        model_name_or_path=cfg.reward_model.model_name_or_path, freeze_backbone=True
    )
    evaluator = Evaluator(
        policy,
        reference,
        reward_model,
        tokenizer,
        max_new_tokens=cfg.ppo.max_new_tokens,
        device=device,
    )
    report = evaluator.evaluate(_load_prompts(Path(prompts)))

    out = cfg.output_dir / "eval_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    logger.info("evaluation report: %s", report.to_dict())


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    typer.run(main)
