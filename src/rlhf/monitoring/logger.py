"""
monitoring.logger — dual W&B / TensorBoard metric logging.

Overview
--------
:class:`RLHFLogger` is a thin, dependency-optional writer. It always keeps an
in-memory history (used by tests and for post-hoc inspection) and *additionally*
mirrors scalars to Weights & Biases and/or TensorBoard when those backends are
configured and importable. Importing this module never imports wandb/tensorboard
eagerly, so it stays side-effect-free.

Usage Example
-------------
>>> from rlhf.monitoring.logger import RLHFLogger
>>> logger = RLHFLogger(run_name="demo")          # in-memory only
>>> logger.log_ppo_step(step=0, metrics={"train/policy_loss": 0.1})
>>> logger.history[0].get("train/policy_loss")
0.1
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class RLHFLogger:
    """
    Multi-backend scalar logger with an always-on in-memory history.

    Args:
        run_name: Human-readable identifier for the run.
        wandb_project: If set and wandb is importable, mirror scalars to W&B.
        tensorboard_dir: If set and tensorboard is importable, write event files.
        config: Optional run config dict recorded at init.
    """

    def __init__(
        self,
        run_name: str = "rlhf-run",
        wandb_project: str | None = None,
        tensorboard_dir: str | Path | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.run_name = run_name
        self.history: list[dict[str, float]] = []
        # Backends are dynamically-typed optional handles (wandb Run / TB writer).
        self._wandb: Any = None
        self._tb: Any = None
        self._init_wandb(wandb_project, config)
        self._init_tensorboard(tensorboard_dir)

    def _init_wandb(self, project: str | None, config: dict[str, Any] | None) -> None:
        if project is None:
            return
        try:
            import wandb

            self._wandb = wandb.init(project=project, name=self.run_name, config=config)
        except Exception as exc:  # pragma: no cover - environment dependent
            # W&B is best-effort telemetry; a failure must never crash training.
            logger.warning("W&B disabled (%s): %s", type(exc).__name__, exc)
            self._wandb = None

    def _init_tensorboard(self, tb_dir: str | Path | None) -> None:
        if tb_dir is None:
            return
        try:
            from torch.utils.tensorboard import SummaryWriter  # type: ignore[attr-defined]

            self._tb = SummaryWriter(log_dir=str(tb_dir))  # type: ignore[no-untyped-call]
        except Exception as exc:  # pragma: no cover - environment dependent
            logger.warning("TensorBoard disabled (%s): %s", type(exc).__name__, exc)
            self._tb = None

    def log_config(self, config: dict[str, Any]) -> None:
        """Record the full run configuration (logged at INFO)."""
        logger.info("run=%s config=%s", self.run_name, config)
        if self._wandb is not None:  # pragma: no cover - requires live wandb
            self._wandb.config.update(config, allow_val_change=True)

    def log_ppo_step(self, step: int, metrics: dict[str, float]) -> None:
        """Log one PPO step's metrics to every configured backend."""
        record = {"step": float(step), **{k: float(v) for k, v in metrics.items()}}
        self.history.append(record)
        logger.debug("step=%d metrics=%s", step, metrics)
        if self._wandb is not None:  # pragma: no cover - requires live wandb
            self._wandb.log(metrics, step=step)
        if self._tb is not None:
            for key, value in metrics.items():
                self._tb.add_scalar(key, float(value), step)

    def log_scalars(self, step: int, metrics: dict[str, float], prefix: str = "") -> None:
        """Log an arbitrary scalar dict, optionally namespaced with ``prefix``."""
        prefixed = {f"{prefix}{k}": v for k, v in metrics.items()}
        self.log_ppo_step(step, prefixed)

    def finish(self) -> None:
        """Flush and close every backend."""
        if self._tb is not None:
            self._tb.flush()
            self._tb.close()
        if self._wandb is not None:  # pragma: no cover - requires live wandb
            self._wandb.finish()


__all__ = ["RLHFLogger"]
