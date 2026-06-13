"""
exceptions — typed exception hierarchy for the RLHF-PPO package.

Overview
--------
Every recoverable failure in the pipeline raises a descriptive, typed exception
derived from :class:`RLHFError`. Bare ``ValueError`` / ``RuntimeError`` are never
raised from library code so that callers can handle failures by category
(data, model, training, security) rather than by string-matching messages.

The hierarchy is intentionally shallow and flat:

    RLHFError
    ├── ConfigError
    ├── DataValidationError
    ├── RolloutError
    │   └── BufferFullError
    ├── RewardModelError
    ├── PolicyModelError
    ├── KLControllerError
    ├── TrainingError
    │   ├── NumericalInstabilityError
    │   ├── RewardHackingDetected
    │   └── KLDivergenceExceeded
    ├── CheckpointError
    │   └── CheckpointTamperingError
    └── SecurityError
        └── PromptInjectionError

Usage Example
-------------
>>> from rlhf.exceptions import BufferFullError
>>> raise BufferFullError("buffer holds 64/64 rollouts")
Traceback (most recent call last):
    ...
rlhf.exceptions.BufferFullError: buffer holds 64/64 rollouts
"""

from __future__ import annotations


class RLHFError(Exception):
    """Base class for every error raised by the RLHF-PPO package."""


class ConfigError(RLHFError):
    """Raised when configuration is internally inconsistent or invalid."""


class DataValidationError(RLHFError):
    """Raised when preference / prompt / rollout data fails validation."""


class RolloutError(RLHFError):
    """Raised for inconsistencies in rollout collection or advantage estimation."""


class BufferFullError(RolloutError):
    """Raised when pushing into a :class:`RolloutBuffer` that is at capacity."""


class RewardModelError(RLHFError):
    """Raised for reward-model construction, training, or scoring failures."""


class PolicyModelError(RLHFError):
    """Raised for policy-model construction, generation, or scoring failures."""


class KLControllerError(RLHFError):
    """Raised when a KL controller is mis-configured or receives invalid input."""


class TrainingError(RLHFError):
    """Base class for failures that occur inside a training loop."""


class NumericalInstabilityError(TrainingError):
    """Raised when a NaN or Inf is detected in a loss or gradient term."""


class RewardHackingDetected(TrainingError):
    """Raised when reward over-optimization crosses the configured threshold."""


class KLDivergenceExceeded(TrainingError):
    """Raised when measured KL exceeds the configured abort threshold."""


class CheckpointError(RLHFError):
    """Raised for checkpoint save / load failures."""


class CheckpointTamperingError(CheckpointError):
    """Raised when a checkpoint's SHA-256 digest does not match its manifest."""


class SecurityError(RLHFError):
    """Base class for security-policy violations."""


class PromptInjectionError(SecurityError):
    """Raised when a prompt matches a known injection pattern in the blocklist."""


__all__ = [
    "BufferFullError",
    "CheckpointError",
    "CheckpointTamperingError",
    "ConfigError",
    "DataValidationError",
    "KLControllerError",
    "KLDivergenceExceeded",
    "NumericalInstabilityError",
    "PolicyModelError",
    "PromptInjectionError",
    "RLHFError",
    "RewardHackingDetected",
    "RewardModelError",
    "RolloutError",
    "SecurityError",
    "TrainingError",
]
