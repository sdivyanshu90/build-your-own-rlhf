"""
config.schema — strongly-typed configuration for the entire RLHF pipeline.

Overview
--------
Every hyperparameter in the pipeline is declared here as a Pydantic field with a
range constraint and a human-readable ``description`` (surfaced in the generated
JSON Schema). The root :class:`RLHFConfig` is a ``pydantic-settings`` model, so
values may be supplied via a YAML file, environment variables (prefix ``RLHF_``),
or direct construction — with full validation in every path.

Cross-field invariants (e.g. ``kl_min < kl_max``, mutually-exclusive 8-bit/4-bit
loading) are enforced with ``model_validator`` hooks so an inconsistent config
fails fast at construction time rather than mid-training.

Usage Example
-------------
>>> from rlhf.config.schema import PPOConfig
>>> cfg = PPOConfig(clip_eps=0.2, kl_min=0.05, kl_max=0.5)
>>> cfg.clip_eps
0.2

References
----------
- Ouyang et al. (2022). Training language models to follow instructions with
  human feedback. https://arxiv.org/abs/2203.02155
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from rlhf.exceptions import ConfigError


class DType(str, Enum):
    """Supported torch weight dtypes, expressed as strings for serialization."""

    FLOAT32 = "float32"
    BFLOAT16 = "bfloat16"
    FLOAT16 = "float16"


class ModelConfig(BaseModel):
    """Configuration for the policy / reference backbone."""

    model_config = {"extra": "forbid"}

    model_name_or_path: str = Field(
        ...,
        description="HuggingFace model id or local path for the causal LM backbone.",
    )
    value_head_dropout: float = Field(
        0.1, ge=0.0, le=1.0, description="Dropout probability inside the value head."
    )
    freeze_layers: int = Field(
        0, ge=0, description="Number of transformer layers to freeze from the bottom."
    )
    dtype: Literal["float32", "bfloat16", "float16"] = Field(
        "bfloat16", description="Torch dtype for model weights."
    )
    load_in_8bit: bool = Field(False, description="Load the backbone in 8-bit precision.")
    load_in_4bit: bool = Field(False, description="Load the backbone in 4-bit precision.")

    @model_validator(mode="after")
    def _validate_quantization(self) -> ModelConfig:
        # 8-bit and 4-bit loading are mutually exclusive — bitsandbytes cannot
        # quantize the same weights to two precisions simultaneously.
        if self.load_in_8bit and self.load_in_4bit:
            raise ConfigError("load_in_8bit and load_in_4bit are mutually exclusive.")
        return self


class PPOConfig(BaseModel):
    """Hyperparameters for the PPO training loop (Section 2.1-2.3 of the spec)."""

    model_config = {"extra": "forbid"}

    # --- Core surrogate-objective hyperparameters ---
    clip_eps: float = Field(
        0.2, gt=0.0, lt=1.0, description="PPO clip range epsilon for the policy ratio."
    )
    clip_eps_vf: float = Field(
        0.2, gt=0.0, description="Clip range for value-function predictions."
    )
    entropy_coeff: float = Field(0.01, ge=0.0, description="Coefficient c2 on the entropy bonus.")
    value_coeff: float = Field(
        0.5, ge=0.0, description="Coefficient c1 on the value-function loss."
    )
    gamma: float = Field(1.0, ge=0.0, le=1.0, description="GAE discount factor gamma.")
    lam: float = Field(0.95, ge=0.0, le=1.0, description="GAE smoothing parameter lambda.")

    # --- Training loop sizing ---
    total_steps: int = Field(1000, gt=0, description="Total number of PPO global steps.")
    rollout_batch_size: int = Field(
        64, gt=0, description="Number of prompts sampled per rollout phase."
    )
    ppo_epochs: int = Field(4, gt=0, description="Optimization epochs over each rollout buffer.")
    mini_batch_size: int = Field(8, gt=0, description="Mini-batch size for each PPO gradient step.")
    max_grad_norm: float = Field(1.0, gt=0.0, description="Gradient clipping max-norm.")
    gradient_accumulation_steps: int = Field(
        1, gt=0, description="Micro-batches accumulated before each optimizer step."
    )

    # --- KL control ---
    kl_target: float = Field(6.0, gt=0.0, description="Target KL(pi || pi_ref) in nats.")
    kl_init: float = Field(0.2, gt=0.0, description="Initial KL penalty coefficient beta.")
    kl_min: float = Field(0.05, gt=0.0, description="Lower clip bound for beta.")
    kl_max: float = Field(0.5, gt=0.0, description="Upper clip bound for beta.")
    kl_step_size: float = Field(
        0.1, gt=0.0, description="Adaptive controller step size alpha (Section 2.3)."
    )
    kl_adaptive: bool = Field(True, description="Use the adaptive (vs fixed) KL controller.")
    kl_abort_threshold: float = Field(
        30.0, gt=0.0, description="Abort training if measured KL exceeds this value."
    )

    # --- Generation ---
    max_new_tokens: int = Field(256, gt=0, description="Max response tokens to generate.")
    temperature: float = Field(1.0, gt=0.0, description="Sampling temperature.")
    top_p: float = Field(1.0, gt=0.0, le=1.0, description="Nucleus sampling top-p.")
    top_k: int = Field(0, ge=0, description="Top-k truncation (0 disables top-k).")
    repetition_penalty: float = Field(
        1.0, ge=1.0, description="Repetition penalty applied during generation."
    )

    # --- Safety / early stopping ---
    reward_hacking_threshold: float = Field(
        0.5, gt=0.0, description="Abort if reward_std / |reward_mean| exceeds this."
    )

    # --- Optimization ---
    learning_rate: float = Field(1.4e-5, gt=0.0, description="AdamW learning rate.")
    lr_scheduler: Literal["cosine", "linear", "constant"] = Field(
        "cosine", description="Learning-rate schedule family."
    )
    warmup_steps: int = Field(100, ge=0, description="Linear warmup steps.")

    # --- Infrastructure ---
    eval_every: int = Field(50, gt=0, description="Run evaluation every N global steps.")
    save_every: int = Field(100, gt=0, description="Checkpoint every N global steps.")
    seed: int = Field(42, description="Random seed for reproducibility.")

    @model_validator(mode="after")
    def _validate_kl_bounds(self) -> PPOConfig:
        # The adaptive controller clips beta into [kl_min, kl_max]; an inverted
        # interval would make the clip a no-op silently, so reject it up front.
        if self.kl_min >= self.kl_max:
            raise ConfigError(
                f"kl_min ({self.kl_min}) must be strictly less than kl_max ({self.kl_max})."
            )
        if not (self.kl_min <= self.kl_init <= self.kl_max):
            raise ConfigError(
                f"kl_init ({self.kl_init}) must lie within [kl_min, kl_max] "
                f"= [{self.kl_min}, {self.kl_max}]."
            )
        # mini_batch_size cannot exceed the rollout batch or some prompts would
        # never appear in a mini-batch.
        if self.mini_batch_size > self.rollout_batch_size:
            raise ConfigError(
                f"mini_batch_size ({self.mini_batch_size}) cannot exceed "
                f"rollout_batch_size ({self.rollout_batch_size})."
            )
        return self


class RewardModelConfig(BaseModel):
    """Configuration for Bradley-Terry reward-model training."""

    model_config = {"extra": "forbid"}

    model_name_or_path: str = Field(..., description="Backbone id/path for the reward model.")
    freeze_backbone: bool = Field(
        True, description="Freeze the backbone and train only the reward head."
    )
    learning_rate: float = Field(1e-5, gt=0.0, description="AdamW learning rate.")
    batch_size: int = Field(32, gt=0, description="Preference pairs per training batch.")
    epochs: int = Field(3, gt=0, description="Number of passes over the preference set.")
    ensemble_size: int = Field(
        1, ge=1, description="Number of independently-initialized reward models."
    )
    normalize_rewards: bool = Field(
        True, description="Apply running mean/variance normalization to rewards."
    )
    max_grad_norm: float = Field(1.0, gt=0.0, description="Gradient clipping max-norm.")
    seed: int = Field(42, description="Random seed.")


class SFTConfig(BaseModel):
    """Configuration for the supervised fine-tuning warmup stage (stage 0)."""

    model_config = {"extra": "forbid"}

    model_name_or_path: str = Field(..., description="Backbone id/path to fine-tune.")
    learning_rate: float = Field(2e-5, gt=0.0, description="AdamW learning rate.")
    batch_size: int = Field(16, gt=0, description="Examples per training batch.")
    epochs: int = Field(1, gt=0, description="Number of passes over the SFT corpus.")
    max_length: int = Field(512, gt=0, description="Max sequence length (tokens).")
    max_grad_norm: float = Field(1.0, gt=0.0, description="Gradient clipping max-norm.")
    warmup_steps: int = Field(0, ge=0, description="Linear warmup steps.")
    seed: int = Field(42, description="Random seed.")


class SecurityConfig(BaseModel):
    """Configuration for input-sanitization and prompt-injection defenses."""

    model_config = {"extra": "forbid"}

    max_prompt_length: int = Field(
        2048, gt=0, description="Max prompt length in tokens before truncation."
    )
    injection_blocklist: list[str] = Field(
        default_factory=lambda: [
            "ignore previous instructions",
            "system:",
            "<|im_start|>",
        ],
        description="Case-insensitive substrings that trigger PromptInjectionError.",
    )
    audit_log_path: Path = Field(
        Path("audit.log"), description="Path to the append-only audit log."
    )


class KLConfig(BaseModel):
    """Self-contained config consumed by :func:`make_kl_controller`."""

    model_config = {"extra": "forbid"}

    adaptive: bool = Field(True, description="Adaptive vs fixed KL coefficient.")
    init_coef: float = Field(0.2, gt=0.0, description="Initial beta value.")
    target: float = Field(6.0, gt=0.0, description="Target KL in nats (adaptive only).")
    step_size: float = Field(0.1, gt=0.0, description="Adaptive step size alpha.")
    coef_min: float = Field(0.05, gt=0.0, description="Lower clip bound for beta.")
    coef_max: float = Field(0.5, gt=0.0, description="Upper clip bound for beta.")

    @model_validator(mode="after")
    def _validate_bounds(self) -> KLConfig:
        if self.coef_min >= self.coef_max:
            raise ConfigError("coef_min must be strictly less than coef_max.")
        return self

    @classmethod
    def from_ppo_config(cls, ppo: PPOConfig) -> KLConfig:
        """Derive a :class:`KLConfig` from the flat fields of a :class:`PPOConfig`."""
        return cls(
            adaptive=ppo.kl_adaptive,
            init_coef=ppo.kl_init,
            target=ppo.kl_target,
            step_size=ppo.kl_step_size,
            coef_min=ppo.kl_min,
            coef_max=ppo.kl_max,
        )


class RLHFConfig(BaseSettings):
    """Root configuration aggregating every sub-config for a full pipeline run."""

    model_config = SettingsConfigDict(
        env_prefix="RLHF_",
        env_nested_delimiter="__",
        extra="forbid",
    )

    model: ModelConfig
    ppo: PPOConfig = Field(default_factory=PPOConfig)
    reward_model: RewardModelConfig
    sft: SFTConfig | None = None
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    output_dir: Path = Field(Path("outputs"), description="Root directory for artifacts.")
    run_name: str = Field("rlhf-run", description="Human-readable run identifier.")
    wandb_project: str | None = Field(None, description="W&B project name (None disables W&B).")
    device: str = Field("cpu", description="Torch device string, e.g. 'cpu' or 'cuda'.")
    seed: int = Field(42, description="Global random seed.")

    @classmethod
    def from_yaml(cls, path: str | Path) -> RLHFConfig:
        """
        Construct a config from a YAML file.

        Args:
            path: Path to a YAML document whose top-level keys mirror the fields
                of :class:`RLHFConfig`.

        Returns:
            A fully-validated :class:`RLHFConfig`.

        Raises:
            ConfigError: If the file is missing, unparseable, or fails validation.
        """
        path = Path(path)
        if not path.is_file():
            raise ConfigError(f"Config file not found: {path}")
        try:
            with path.open("r", encoding="utf-8") as fh:
                raw: dict[str, Any] = yaml.safe_load(fh) or {}
        except yaml.YAMLError as exc:  # pragma: no cover - exercised via tests
            raise ConfigError(f"Failed to parse YAML config {path}: {exc}") from exc
        if not isinstance(raw, dict):
            raise ConfigError(f"Top-level YAML in {path} must be a mapping.")
        return cls(**raw)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict of the full config (for run logging)."""
        return self.model_dump(mode="json")


__all__ = [
    "DType",
    "KLConfig",
    "ModelConfig",
    "PPOConfig",
    "RLHFConfig",
    "RewardModelConfig",
    "SFTConfig",
    "SecurityConfig",
]
