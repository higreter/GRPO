"""Reward package: pure scoring logic plus thin TRL adapters."""

from __future__ import annotations

from .base import REASON_CORRECT, REASON_INCORRECT, REASON_INVALID_OUTPUT, Outcome, invalid, scored

__all__ = [
    "REASON_CORRECT",
    "REASON_INCORRECT",
    "REASON_INVALID_OUTPUT",
    "Outcome",
    "invalid",
    "scored",
]
