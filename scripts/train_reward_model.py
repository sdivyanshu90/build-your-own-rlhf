#!/usr/bin/env python3
"""
train_reward_model.py — reward-model training entrypoint (RLHF stage 1).

Loads an RLHF config and a JSON-lines preference dataset, then trains either a
single reward model or an ensemble (per ``reward_model.ensemble_size``) with the
Bradley-Terry objective.

Usage:
    python scripts/train_reward_model.py --config config.yaml --data prefs.jsonl
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
import typer
from transformers import AutoTokenizer

from rlhf.config.schema import RLHFConfig
from rlhf.data.schemas import Preference
from rlhf.models.reward_model import RewardModel, RewardModelEnsemble
from rlhf.monitoring.logger import RLHFLogger
from rlhf.security.audit import CheckpointVerifier
from rlhf.training.reward_model.ensemble import EnsembleRewardModelTrainer
from rlhf.training.reward_model.trainer import RewardModelTrainer
from rlhf.utils import reproducibility_info, set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("rlhf.scripts.reward")


def _load_preferences(path: Path) -> list[Preference]:
    """Read a JSON-lines preference dataset into validated Preference objects."""
    prefs: list[Preference] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                prefs.append(Preference(**json.loads(line)))
    return prefs


def main(config: str, data: str = "prefs.jsonl") -> None:
    """Train a reward model (or ensemble) from a config and JSONL preferences."""
    cfg = RLHFConfig.from_yaml(config)
    set_seed(cfg.reward_model.seed, deterministic=True)
    logger.info("reproducibility: %s", reproducibility_info(cfg.to_dict()))

    device = torch.device(cfg.device)
    tokenizer = AutoTokenizer.from_pretrained(cfg.reward_model.model_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    preferences = _load_preferences(Path(data))
    run_logger = RLHFLogger(run_name=f"{cfg.run_name}-rm", wandb_project=cfg.wandb_project)

    if cfg.reward_model.ensemble_size > 1:
        ensemble = RewardModelEnsemble.from_config(
            _backbone_config(cfg.reward_model.model_name_or_path),
            ensemble_size=cfg.reward_model.ensemble_size,
            freeze_backbone=cfg.reward_model.freeze_backbone,
        )
        EnsembleRewardModelTrainer(ensemble, tokenizer, cfg.reward_model, device, run_logger).train(
            preferences
        )
        state = ensemble.state_dict()
    else:
        reward_model = RewardModel(
            model_name_or_path=cfg.reward_model.model_name_or_path,
            freeze_backbone=cfg.reward_model.freeze_backbone,
            normalize_rewards=cfg.reward_model.normalize_rewards,
        )
        RewardModelTrainer(reward_model, tokenizer, cfg.reward_model, device, run_logger).train(
            preferences
        )
        state = reward_model.state_dict()

    out_dir = cfg.output_dir / "reward_model"
    CheckpointVerifier().save(state, out_dir / "reward_model.pt")
    tokenizer.save_pretrained(out_dir)
    run_logger.finish()
    logger.info("reward-model training complete; saved to %s", out_dir)


def _backbone_config(name_or_path: str):  # type: ignore[no-untyped-def]
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(name_or_path)


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    typer.run(main)
