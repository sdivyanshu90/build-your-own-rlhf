"""Unit tests for the KL penalty controllers."""

from __future__ import annotations

import pytest

from rlhf.config.schema import KLConfig
from rlhf.exceptions import KLControllerError
from rlhf.training.ppo.kl_controller import (
    AdaptiveKLController,
    FixedKLController,
    make_kl_controller,
)


def _adaptive() -> AdaptiveKLController:
    return AdaptiveKLController(
        init_coef=0.2, target=6.0, step_size=0.05, coef_min=0.05, coef_max=0.5
    )


def test_adaptive_increases_beta_when_kl_above_target() -> None:
    ctrl = _adaptive()
    before = ctrl.value
    ctrl.update(kl_measured=8.0)  # above target 6.0
    assert ctrl.value > before


def test_adaptive_decreases_beta_when_kl_below_target() -> None:
    ctrl = _adaptive()
    before = ctrl.value
    ctrl.update(kl_measured=2.0)  # below target 6.0
    assert ctrl.value < before


def test_adaptive_clips_beta_to_bounds() -> None:
    ctrl = _adaptive()
    for _ in range(1000):
        ctrl.update(kl_measured=1e6)  # drive far above target
    assert ctrl.value == pytest.approx(0.5)  # clipped at coef_max
    for _ in range(1000):
        ctrl.update(kl_measured=-1e6)  # drive far below target
    assert ctrl.value == pytest.approx(0.05)  # clipped at coef_min


def test_fixed_controller_never_changes() -> None:
    ctrl = FixedKLController(coef=0.3)
    for kl in (0.0, 6.0, 1000.0, -50.0):
        ctrl.update(kl_measured=kl)
        assert ctrl.value == pytest.approx(0.3)


def test_factory_returns_correct_type() -> None:
    adaptive = make_kl_controller(KLConfig(adaptive=True, init_coef=0.2, target=6.0))
    fixed = make_kl_controller(KLConfig(adaptive=False, init_coef=0.2, target=6.0))
    assert isinstance(adaptive, AdaptiveKLController)
    assert isinstance(fixed, FixedKLController)


def test_state_dict_round_trip() -> None:
    ctrl = _adaptive()
    ctrl.update(kl_measured=9.0)
    state = ctrl.state_dict()
    restored = _adaptive()
    restored.load_state_dict(state)
    assert restored.value == pytest.approx(ctrl.value)


def test_invalid_inputs_raise() -> None:
    with pytest.raises(KLControllerError):
        FixedKLController(coef=-1.0)
    with pytest.raises(KLControllerError):
        AdaptiveKLController(init_coef=0.2, target=6.0, step_size=0.1, coef_min=0.5, coef_max=0.05)
    with pytest.raises(KLControllerError):
        _adaptive().update(float("nan"))
