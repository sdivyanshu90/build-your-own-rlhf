"""
security.validation — prompt sanitization and injection defenses.

Overview
--------
:func:`sanitize_prompt` is the single choke-point every externally-sourced prompt
passes through before it reaches a tokenizer or model. It:

1. strips null bytes and control characters (keeping ``\\n`` / ``\\t``),
2. raises :class:`~rlhf.exceptions.PromptInjectionError` when the prompt matches a
   configurable blocklist of known injection patterns, and
3. truncates the prompt to ``config.max_prompt_length`` tokens.

The function never mutates its input (returns a new string) so callers may keep
the original for auditing.

Usage Example
-------------
>>> from rlhf.config.schema import SecurityConfig
>>> from rlhf.security.validation import sanitize_prompt
>>> sanitize_prompt("hello\\x00world", SecurityConfig(), tokenizer=None)
'helloworld'
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rlhf.config.schema import SecurityConfig
from rlhf.exceptions import PromptInjectionError

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

# Control characters that are stripped. We retain newline (\n=10) and tab (\t=9)
# because they carry legitimate formatting; everything else in the C0 range plus
# DEL is removed.
_ALLOWED_CONTROL = {9, 10}
_STRIPPED_ORDINALS = frozenset([c for c in range(0x20) if c not in _ALLOWED_CONTROL] + [0x7F])


def _strip_control_chars(text: str) -> str:
    """Remove null bytes and C0/DEL control characters except tab and newline."""
    return "".join(ch for ch in text if ord(ch) not in _STRIPPED_ORDINALS)


def _detect_injection(text: str, blocklist: list[str]) -> str | None:
    """Return the first blocklist pattern found in ``text`` (case-insensitive)."""
    lowered = text.lower()
    for pattern in blocklist:
        if pattern.lower() in lowered:
            return pattern
    return None


def sanitize_prompt(
    prompt: str,
    config: SecurityConfig,
    tokenizer: PreTrainedTokenizerBase | None = None,
) -> str:
    """
    Sanitize an untrusted prompt.

    Args:
        prompt: The raw prompt string.
        config: Security policy (blocklist, max length).
        tokenizer: Optional tokenizer; when provided, truncation is by *tokens*
            rather than characters.

    Returns:
        A new, sanitized prompt string (the input is never mutated).

    Raises:
        PromptInjectionError: If the prompt matches a blocklist pattern.
    """
    cleaned = _strip_control_chars(prompt)

    pattern = _detect_injection(cleaned, config.injection_blocklist)
    if pattern is not None:
        raise PromptInjectionError(f"prompt matched injection pattern {pattern!r}; rejected.")

    # Truncate to the configured budget. Prefer token-accurate truncation when a
    # tokenizer is available; fall back to a character cap otherwise.
    if tokenizer is not None:
        ids = tokenizer.encode(cleaned, add_special_tokens=False)
        if len(ids) > config.max_prompt_length:
            ids = ids[: config.max_prompt_length]
            cleaned = tokenizer.decode(ids, skip_special_tokens=True)
    elif len(cleaned) > config.max_prompt_length:
        cleaned = cleaned[: config.max_prompt_length]
    return cleaned


__all__ = ["sanitize_prompt"]
