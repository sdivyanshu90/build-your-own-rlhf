"""Unit tests for monitoring.logger and monitoring.alerts."""

from __future__ import annotations

import math

import pytest

from rlhf.exceptions import (
    KLDivergenceExceeded,
    NumericalInstabilityError,
    RewardHackingDetected,
    TrainingError,
)
from rlhf.monitoring.alerts import AlertCondition, AlertManager
from rlhf.monitoring.logger import RLHFLogger


def test_logger_in_memory_history() -> None:
    logger = RLHFLogger(run_name="t")
    logger.log_ppo_step(0, {"train/loss": 1.0})
    logger.log_ppo_step(1, {"train/loss": 0.5})
    assert len(logger.history) == 2
    assert logger.history[0]["train/loss"] == 1.0
    assert logger.history[1]["step"] == 1.0


def test_logger_scalars_prefix_and_config() -> None:
    logger = RLHFLogger(run_name="t")
    logger.log_scalars(0, {"acc": 0.9}, prefix="eval/")
    assert logger.history[0]["eval/acc"] == 0.9
    logger.log_config({"lr": 1e-4})  # must not raise
    logger.finish()


def test_logger_tensorboard_backend(tmp_path) -> None:  # type: ignore[no-untyped-def]
    logger = RLHFLogger(run_name="t", tensorboard_dir=tmp_path / "tb")
    logger.log_ppo_step(0, {"train/loss": 1.0})
    logger.finish()
    # TensorBoard wrote at least one event file.
    assert any((tmp_path / "tb").iterdir())


def test_alert_manager_all_healthy() -> None:
    mgr = AlertManager()
    events = mgr.check({"train/kl_divergence": 1.0, "train/entropy": 2.0})
    assert events == []


def test_alert_kl_spike_raises() -> None:
    mgr = AlertManager(kl_abort_threshold=10.0)
    with pytest.raises(KLDivergenceExceeded):
        mgr.check({"train/kl_divergence": 50.0})


def test_alert_nan_loss_raises() -> None:
    mgr = AlertManager()
    with pytest.raises(NumericalInstabilityError):
        mgr.check({"train/total_loss": math.nan})


def test_alert_reward_hacking_raises() -> None:
    mgr = AlertManager(reward_hacking_threshold=0.5)
    with pytest.raises(RewardHackingDetected):
        mgr.check({"reward/hacking_score": 0.9})


def test_alert_gradient_explosion_raises() -> None:
    mgr = AlertManager(gradient_norm_ceiling=100.0)
    with pytest.raises(NumericalInstabilityError):
        mgr.check({"train/gradient_norm": 500.0})


def test_alert_entropy_collapse_and_value_divergence() -> None:
    mgr = AlertManager()
    with pytest.raises(TrainingError):
        mgr.check({"train/entropy": 0.01})
    with pytest.raises(TrainingError):
        mgr.check({"train/explained_variance": -5.0})


def test_alert_no_raise_mode_collects_events_and_webhook() -> None:
    seen: list[str] = []
    mgr = AlertManager(raise_on_trigger=False, webhook=seen.append)
    events = mgr.check({"train/kl_divergence": 1e6, "train/gradient_norm": 1e6})
    conditions = {e.condition for e in events}
    assert AlertCondition.KL_DIVERGENCE_SPIKE in conditions
    assert AlertCondition.GRADIENT_EXPLOSION in conditions
    assert len(seen) == len(events)


def test_alert_webhook_failure_is_swallowed() -> None:
    def bad_hook(_msg: str) -> None:
        raise RuntimeError("webhook down")

    mgr = AlertManager(raise_on_trigger=False, webhook=bad_hook)
    # A failing webhook must not propagate.
    mgr.check({"train/kl_divergence": 1e6})
