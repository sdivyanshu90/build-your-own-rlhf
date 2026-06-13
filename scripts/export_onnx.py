#!/usr/bin/env python3
"""
export_onnx.py — export a trained policy backbone to ONNX for serving.

Exports the causal-LM backbone (without the value head, which is only needed
during training) to an ONNX graph with dynamic batch / sequence axes.

Usage:
    python scripts/export_onnx.py --config config.yaml --output policy.onnx
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import typer
from transformers import AutoTokenizer

from rlhf.config.schema import RLHFConfig
from rlhf.models.policy import PolicyModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("rlhf.scripts.export")


def main(config: str, output: str = "policy.onnx", opset: int = 17) -> None:
    """Export the policy backbone to ONNX with dynamic axes."""
    cfg = RLHFConfig.from_yaml(config)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    policy = PolicyModel(model_name_or_path=cfg.model.model_name_or_path)
    policy.eval()

    sample = tokenizer("Hello world", return_tensors="pt")
    dynamic_axes = {
        "input_ids": {0: "batch", 1: "sequence"},
        "attention_mask": {0: "batch", 1: "sequence"},
        "logits": {0: "batch", 1: "sequence"},
    }
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        policy.backbone,
        (sample["input_ids"], sample["attention_mask"]),
        str(out_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        opset_version=opset,
    )
    logger.info("exported ONNX policy to %s", out_path)


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    typer.run(main)
