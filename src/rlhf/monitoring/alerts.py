"""
monitoring.alerts — training anomaly detection and alerting.

Overview
--------
:class:`AlertManager` evaluates a small set of training-health conditions every
PPO step. When a condition fires it (1) logs at ERROR, (2) invokes an optional
webhook (Slack / PagerDuty), and (3) raises the mapped typed exception so the
trainer halts *after* its current checkpoint is safe — preserving the last good
model rather than letting a diverging run overwrite it.

The conditions mirror the spec:

============================  ==========================================
Condition                     Trigger
============================  ==========================================
KL_DIVERGENCE_SPIKE           mean KL > kl_abort_threshold
REWARD_HACKING                reward_hacking_score > threshold
VALUE_DIVERGENCE              explained_variance < value_divergence_floor
NAN_IN_LOSS                   any tracked loss is NaN/Inf
ENTROPY_COLLAPSE              entropy < entropy_floor
GRADIENT_EXPLOSION            gradient_norm > gradient_norm_ceiling
============================  ==========================================

Usage Example
-------------
>>> from rlhf.monitoring.alerts import AlertManager
>>> mgr = AlertManager()
>>> mgr.check({"train/kl_divergence": 0.5, "train/entropy": 2.0})  # all healthy
[]
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from rlhf.exceptions import (
    KLDivergenceExceeded,
    NumericalInstabilityError,
    RewardHackingDetected,
    TrainingError,
)

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Severity of a fired alert."""

    INFO = "info"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertCondition(Enum):
    """The set of monitored training-health conditions."""

    KL_DIVERGENCE_SPIKE = "kl > kl_abort_threshold"
    REWARD_HACKING = "reward_hacking_score > threshold"
    VALUE_DIVERGENCE = "explained_variance < floor"
    NAN_IN_LOSS = "any loss is NaN"
    ENTROPY_COLLAPSE = "entropy < floor"
    GRADIENT_EXPLOSION = "gradient_norm > ceiling"


@dataclass(frozen=True)
class AlertEvent:
    """A fired alert: which condition, how severe, and a human message."""

    condition: AlertCondition
    level: AlertLevel
    message: str


# Map each condition to the exception the trainer should raise on trigger.
_CONDITION_EXCEPTION: dict[AlertCondition, type[TrainingError]] = {
    AlertCondition.KL_DIVERGENCE_SPIKE: KLDivergenceExceeded,
    AlertCondition.REWARD_HACKING: RewardHackingDetected,
    AlertCondition.VALUE_DIVERGENCE: TrainingError,
    AlertCondition.NAN_IN_LOSS: NumericalInstabilityError,
    AlertCondition.ENTROPY_COLLAPSE: TrainingError,
    AlertCondition.GRADIENT_EXPLOSION: NumericalInstabilityError,
}


class AlertManager:
    """
    Evaluates training-health conditions and escalates on trigger.

    Args:
        kl_abort_threshold: Mean KL above which to abort.
        reward_hacking_threshold: Reward-hacking score above which to abort.
        entropy_floor: Entropy below which to flag collapse.
        gradient_norm_ceiling: Gradient norm above which to flag explosion.
        value_divergence_floor: Explained variance below which to flag divergence.
        webhook: Optional callable invoked with the formatted alert string.
        raise_on_trigger: If True (default), :meth:`check` raises after alerting.
    """

    def __init__(
        self,
        kl_abort_threshold: float = 30.0,
        reward_hacking_threshold: float = 0.5,
        entropy_floor: float = 0.1,
        gradient_norm_ceiling: float = 100.0,
        value_divergence_floor: float = -1.0,
        webhook: Callable[[str], None] | None = None,
        raise_on_trigger: bool = True,
    ) -> None:
        self.kl_abort_threshold = kl_abort_threshold
        self.reward_hacking_threshold = reward_hacking_threshold
        self.entropy_floor = entropy_floor
        self.gradient_norm_ceiling = gradient_norm_ceiling
        self.value_divergence_floor = value_divergence_floor
        self.webhook = webhook
        self.raise_on_trigger = raise_on_trigger

    def _evaluate(self, metrics: dict[str, float]) -> list[AlertEvent]:
        """Return every condition that the current metrics trigger (no raising)."""
        events: list[AlertEvent] = []

        def _is_bad(x: float) -> bool:
            return math.isnan(x) or math.isinf(x)

        # NaN/Inf in any tracked loss is the most severe — check it first.
        for key in ("train/total_loss", "train/policy_loss", "train/value_loss"):
            if key in metrics and _is_bad(metrics[key]):
                events.append(
                    AlertEvent(
                        AlertCondition.NAN_IN_LOSS,
                        AlertLevel.CRITICAL,
                        f"{key} is not finite ({metrics[key]}).",
                    )
                )

        kl = metrics.get("train/kl_divergence")
        if kl is not None and kl > self.kl_abort_threshold:
            events.append(
                AlertEvent(
                    AlertCondition.KL_DIVERGENCE_SPIKE,
                    AlertLevel.HIGH,
                    f"mean KL {kl:.3f} exceeds abort threshold {self.kl_abort_threshold}.",
                )
            )

        hack = metrics.get("reward/hacking_score")
        if hack is not None and hack > self.reward_hacking_threshold:
            events.append(
                AlertEvent(
                    AlertCondition.REWARD_HACKING,
                    AlertLevel.HIGH,
                    f"reward-hacking score {hack:.3f} exceeds {self.reward_hacking_threshold}.",
                )
            )

        ev = metrics.get("train/explained_variance")
        if ev is not None and ev < self.value_divergence_floor:
            events.append(
                AlertEvent(
                    AlertCondition.VALUE_DIVERGENCE,
                    AlertLevel.MEDIUM,
                    f"explained variance {ev:.3f} below floor {self.value_divergence_floor}.",
                )
            )

        entropy = metrics.get("train/entropy")
        if entropy is not None and entropy < self.entropy_floor:
            events.append(
                AlertEvent(
                    AlertCondition.ENTROPY_COLLAPSE,
                    AlertLevel.MEDIUM,
                    f"entropy {entropy:.3f} below floor {self.entropy_floor}.",
                )
            )

        grad = metrics.get("train/gradient_norm")
        if grad is not None and grad > self.gradient_norm_ceiling:
            events.append(
                AlertEvent(
                    AlertCondition.GRADIENT_EXPLOSION,
                    AlertLevel.HIGH,
                    f"gradient norm {grad:.3f} exceeds ceiling {self.gradient_norm_ceiling}.",
                )
            )
        return events

    def check(self, metrics: dict[str, float]) -> list[AlertEvent]:
        """
        Evaluate all conditions; log + webhook + (optionally) raise on trigger.

        Returns:
            The list of fired events. If ``raise_on_trigger`` is True and any
            event fired, the highest-priority event's mapped exception is raised
            (after logging and webhook delivery).
        """
        events = self._evaluate(metrics)
        for event in events:
            message = f"[{event.level.value.upper()}] {event.condition.name}: {event.message}"
            logger.error(message)
            if self.webhook is not None:
                try:
                    self.webhook(message)
                except Exception as exc:  # pragma: no cover - webhook best-effort
                    logger.warning("alert webhook failed: %s", exc)
        if events and self.raise_on_trigger:
            primary = events[0]
            raise _CONDITION_EXCEPTION[primary.condition](primary.message)
        return events


__all__ = ["AlertCondition", "AlertEvent", "AlertLevel", "AlertManager"]
