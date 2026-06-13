#!/usr/bin/env python3
"""
train_sft.py — supervised fine-tuning entrypoint (RLHF stage 0).

Loads an RLHF YAML config, instantiates the causal LM and tokenizer, reads an
SFT corpus of ``{"prompt", "completion"}`` JSON-lines, and runs the SFTTrainer.

Usage:
    python scripts/train_sft.py --config config.yaml --data sft.jsonl
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
import typer
from transformers import AutoModelForCausalLM, AutoTokenizer

from rlhf.config.schema import RLHFConfig
from rlhf.exceptions import ConfigError
from rlhf.monitoring.logger import RLHFLogger
from rlhf.training.sft.trainer import SFTTrainer
from rlhf.utils import reproducibility_info, set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("rlhf.scripts.sft")


def _load_examples(path: Path) -> list[dict[str, str]]:
    """Read a JSON-lines SFT corpus of prompt/completion pairs."""
    examples: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def main(config: str, data: str = "sft.jsonl") -> None:
    """Run SFT from a config file and a JSONL data file."""
    cfg = RLHFConfig.from_yaml(config)
    if cfg.sft is None:
        raise ConfigError("config.sft section is required for SFT.")
    set_seed(cfg.seed, deterministic=True)
    logger.info("reproducibility: %s", reproducibility_info(cfg.to_dict()))

    device = torch.device(cfg.device)
    tokenizer = AutoTokenizer.from_pretrained(cfg.sft.model_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg.sft.model_name_or_path)

    run_logger = RLHFLogger(run_name=f"{cfg.run_name}-sft", wandb_project=cfg.wandb_project)
    trainer = SFTTrainer(model, tokenizer, cfg.sft, device=device, logger_backend=run_logger)
    trainer.train(_load_examples(Path(data)))

    out_dir = cfg.output_dir / "sft"
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    run_logger.finish()
    logger.info("SFT complete; model saved to %s", out_dir)


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    typer.run(main)
