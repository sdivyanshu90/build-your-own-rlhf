"""Unit tests for the configuration schema and its validators."""

from __future__ import annotations

import pytest

from rlhf.config.schema import (
    KLConfig,
    ModelConfig,
    PPOConfig,
    RewardModelConfig,
    RLHFConfig,
)
from rlhf.exceptions import ConfigError


def test_ppo_defaults_valid() -> None:
    cfg = PPOConfig()
    assert cfg.clip_eps == 0.2
    assert cfg.kl_min < cfg.kl_max


def test_ppo_rejects_inverted_kl_bounds() -> None:
    with pytest.raises(ConfigError, match="kl_min"):
        PPOConfig(kl_min=0.5, kl_max=0.1)


def test_ppo_rejects_kl_init_out_of_bounds() -> None:
    with pytest.raises(ConfigError, match="kl_init"):
        PPOConfig(kl_min=0.1, kl_max=0.2, kl_init=0.9)


def test_ppo_rejects_oversized_mini_batch() -> None:
    with pytest.raises(ConfigError, match="mini_batch_size"):
        PPOConfig(rollout_batch_size=4, mini_batch_size=8)


def test_model_config_rejects_dual_quantization() -> None:
    with pytest.raises(ConfigError, match="mutually exclusive"):
        ModelConfig(model_name_or_path="gpt2", load_in_8bit=True, load_in_4bit=True)


def test_kl_config_from_ppo() -> None:
    ppo = PPOConfig(kl_adaptive=True, kl_init=0.2, kl_target=6.0)
    kl = KLConfig.from_ppo_config(ppo)
    assert kl.adaptive is True
    assert kl.init_coef == 0.2
    assert kl.coef_max == ppo.kl_max


def test_kl_config_rejects_inverted_bounds() -> None:
    with pytest.raises(ConfigError):
        KLConfig(coef_min=0.5, coef_max=0.1)


def test_rlhf_config_from_yaml(tmp_path) -> None:  # type: ignore[no-untyped-def]
    yaml_text = """
model:
  model_name_or_path: gpt2
reward_model:
  model_name_or_path: gpt2
run_name: my-run
output_dir: /tmp/out
"""
    path = tmp_path / "config.yaml"
    path.write_text(yaml_text)
    cfg = RLHFConfig.from_yaml(path)
    assert cfg.run_name == "my-run"
    assert cfg.model.model_name_or_path == "gpt2"
    assert isinstance(cfg.to_dict(), dict)


def test_rlhf_config_from_yaml_missing_file() -> None:
    with pytest.raises(ConfigError, match="not found"):
        RLHFConfig.from_yaml("/nonexistent/config.yaml")


def test_rlhf_config_rejects_extra_fields() -> None:
    with pytest.raises(Exception, match="extra"):
        RLHFConfig(
            model=ModelConfig(model_name_or_path="gpt2"),
            reward_model=RewardModelConfig(model_name_or_path="gpt2"),
            bogus_field=123,
        )
