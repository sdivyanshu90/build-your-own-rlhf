"""
training.ppo.kl_controller — adaptive and fixed KL penalty controllers.

Overview
--------
The PPO-for-RLHF objective subtracts ``beta * KL(pi_theta || pi_ref)`` from the
reward to keep the policy near the reference model. ``beta`` may either be held
constant (:class:`FixedKLController`) or adapted online toward a target KL
(:class:`AdaptiveKLController`). Both implement the same small interface so the
trainer is agnostic to the choice.

Mathematical Background
-----------------------
Adaptive update (Section 2.3 of the spec)::

    beta <- beta + alpha * (KL_measured - KL_target)
    beta <- clip(beta, beta_min, beta_max)

Usage Example
-------------
>>> from rlhf.config.schema import KLConfig
>>> from rlhf.training.ppo.kl_controller import make_kl_controller
>>> ctrl = make_kl_controller(KLConfig(adaptive=True, init_coef=0.2, target=6.0))
>>> ctrl.update(kl_measured=10.0)   # KL above target -> beta rises
>>> ctrl.value > 0.2
True

References
----------
- Ziegler et al. (2019). Fine-Tuning Language Models from Human Preferences.
  https://arxiv.org/abs/1909.08593
"""

from __future__ import annotations

import abc
import math

from rlhf.config.schema import KLConfig
from rlhf.exceptions import KLControllerError


class AbstractKLController(abc.ABC):
    """Interface shared by every KL penalty controller."""

    @property
    @abc.abstractmethod
    def value(self) -> float:
        """Current KL penalty coefficient ``beta``."""

    @abc.abstractmethod
    def update(self, kl_measured: float) -> None:
        """Update ``beta`` given the most recently measured mean KL."""

    @abc.abstractmethod
    def state_dict(self) -> dict[str, float]:
        """Serialize controller state for checkpointing."""

    @abc.abstractmethod
    def load_state_dict(self, state: dict[str, float]) -> None:
        """Restore controller state from :meth:`state_dict`."""


class FixedKLController(AbstractKLController):
    """A KL controller that returns a constant coefficient and never adapts."""

    def __init__(self, coef: float) -> None:
        if coef <= 0:
            raise KLControllerError(f"fixed KL coef must be positive, got {coef}.")
        self._coef = float(coef)

    @property
    def value(self) -> float:
        return self._coef

    def update(self, kl_measured: float) -> None:
        # Intentionally a no-op: the fixed controller ignores measured KL. We
        # still validate the input so callers cannot silently pass NaN/Inf.
        if not math.isfinite(kl_measured):
            raise KLControllerError(f"kl_measured must be finite, got {kl_measured}.")

    def state_dict(self) -> dict[str, float]:
        return {"coef": self._coef}

    def load_state_dict(self, state: dict[str, float]) -> None:
        self._coef = float(state["coef"])


class AdaptiveKLController(AbstractKLController):
    """
    Proportional KL controller that nudges ``beta`` toward a target KL.

    Args:
        init_coef: Initial coefficient ``beta``.
        target: Target mean KL in nats.
        step_size: Proportional gain ``alpha``.
        coef_min: Lower clip bound for ``beta``.
        coef_max: Upper clip bound for ``beta``.
    """

    def __init__(
        self,
        init_coef: float,
        target: float,
        step_size: float,
        coef_min: float,
        coef_max: float,
    ) -> None:
        if coef_min >= coef_max:
            raise KLControllerError("coef_min must be strictly less than coef_max.")
        if not (coef_min <= init_coef <= coef_max):
            raise KLControllerError(
                f"init_coef {init_coef} must lie within [{coef_min}, {coef_max}]."
            )
        self._coef = float(init_coef)
        self._target = float(target)
        self._step_size = float(step_size)
        self._coef_min = float(coef_min)
        self._coef_max = float(coef_max)

    @property
    def value(self) -> float:
        return self._coef

    def update(self, kl_measured: float) -> None:
        """
        Apply one proportional update step toward the target KL.

        When the measured KL exceeds the target the coefficient rises (tightening
        the leash on the policy); when it is below target the coefficient falls.
        """
        if not math.isfinite(kl_measured):
            raise KLControllerError(f"kl_measured must be finite, got {kl_measured}.")
        # beta <- beta + alpha * (KL_measured - KL_target), then clip to bounds.
        self._coef += self._step_size * (kl_measured - self._target)
        self._coef = min(max(self._coef, self._coef_min), self._coef_max)

    def state_dict(self) -> dict[str, float]:
        return {
            "coef": self._coef,
            "target": self._target,
            "step_size": self._step_size,
            "coef_min": self._coef_min,
            "coef_max": self._coef_max,
        }

    def load_state_dict(self, state: dict[str, float]) -> None:
        self._coef = float(state["coef"])
        self._target = float(state["target"])
        self._step_size = float(state["step_size"])
        self._coef_min = float(state["coef_min"])
        self._coef_max = float(state["coef_max"])


def make_kl_controller(config: KLConfig) -> AbstractKLController:
    """
    Construct the KL controller selected by ``config.adaptive``.

    Args:
        config: A :class:`~rlhf.config.schema.KLConfig`.

    Returns:
        An :class:`AdaptiveKLController` if ``config.adaptive`` else a
        :class:`FixedKLController`.
    """
    if config.adaptive:
        return AdaptiveKLController(
            init_coef=config.init_coef,
            target=config.target,
            step_size=config.step_size,
            coef_min=config.coef_min,
            coef_max=config.coef_max,
        )
    return FixedKLController(coef=config.init_coef)


__all__ = [
    "AbstractKLController",
    "AdaptiveKLController",
    "FixedKLController",
    "make_kl_controller",
]
