#!/usr/bin/env python3
"""
train_ppo.py — PPO policy-optimization entrypoint (RLHF stage 2).

Loads an RLHF config, the SFT policy and tokenizer, a frozen reference (a copy of
the policy), the trained reward model, and a JSON-lines prompt dataset, then runs
the PPOTrainer with full logging and anomaly alerting.

Usage:
    python scripts/train_ppo.py --config config.yaml --prompts prompts.jsonl
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
import typer
from transformers import AutoTokenizer

from rlhf.config.schema import RLHFConfig
from rlhf.models.policy import PolicyModel
from rlhf.models.reference_model import ReferenceModel
from rlhf.models.reward_model import RewardModel
from rlhf.monitoring.alerts import AlertManager
from rlhf.monitoring.logger import RLHFLogger
from rlhf.security.audit import CheckpointVerifier
from rlhf.utils import reproducibility_info, set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("rlhf.scripts.ppo")


def _load_prompts(path: Path) -> list[str]:
    """Read prompts (one JSON object with a 'text'/'prompt' field per line)."""
    prompts: list[str] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            prompts.append(obj.get("prompt") or obj.get("text") or "")
    return [p for p in prompts if p]


def main(config: str, prompts: str = "prompts.jsonl") -> None:
    """Run PPO from a config file and a JSONL prompt file."""
    from rlhf.training.ppo.trainer import PPOTrainer

    cfg = RLHFConfig.from_yaml(config)
    set_seed(cfg.ppo.seed, deterministic=True)
    logger.info("reproducibility: %s", reproducibility_info(cfg.to_dict()))

    device = torch.device(cfg.device)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    policy = PolicyModel(
        model_name_or_path=cfg.model.model_name_or_path,
        value_head_dropout=cfg.model.value_head_dropout,
        freeze_layers=cfg.model.freeze_layers,
        dtype=cfg.model.dtype,
    )
    reference = ReferenceModel.from_policy(policy)
    reward_model = RewardModel(
        model_name_or_path=cfg.reward_model.model_name_or_path,
        freeze_backbone=True,
        normalize_rewards=cfg.reward_model.normalize_rewards,
    )

    run_logger = RLHFLogger(
        run_name=f"{cfg.run_name}-ppo",
        wandb_project=cfg.wandb_project,
        tensorboard_dir=cfg.output_dir / "tb",
        config=cfg.to_dict(),
    )
    alert_manager = AlertManager(
        kl_abort_threshold=cfg.ppo.kl_abort_threshold,
        reward_hacking_threshold=cfg.ppo.reward_hacking_threshold,
    )
    trainer = PPOTrainer(
        policy,
        reference,
        tokenizer,
        cfg.ppo,
        _load_prompts(Path(prompts)),
        reward_model=reward_model,
        device=device,
        logger_backend=run_logger,
        alert_manager=alert_manager,
        output_dir=cfg.output_dir / "ppo",
    )
    try:
        trainer.train()
    finally:
        # Always persist the final (or last-good) policy with an integrity manifest.
        CheckpointVerifier().save(
            trainer.state_dict(),
            cfg.output_dir / "ppo" / "final.pt",
            step=trainer.global_step,
        )
        run_logger.finish()
    logger.info("PPO training complete at step %d", trainer.global_step)


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    typer.run(main)
