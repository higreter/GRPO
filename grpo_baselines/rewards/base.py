"""Reward functions for the four baselines.

Each module exposes pure functions that take a parsed completion and the ground truth and
return an :class:`Outcome` -- ``(reward, reason, detail)`` -- so the same code path feeds
both the trainer and the evaluation JSONL.  Thin TRL adapters (``make_*_reward_fn``) wrap
those pure functions without re-implementing any logic.

Why ``Outcome`` instead of a bare float: the task requires that invalid outputs, wrong
verdicts and correct verdicts be distinguishable *after the fact*.  A float alone cannot
tell "wrong answer" from "unparsable answer" once it has been averaged into a batch, and
the report has to state parser-success and invalid-output rates separately.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

__all__ = ["Outcome", "invalid", "scored"]

# Reason codes shared by the reward layer (the parser contributes its own, see
# common.parsing.REASON_CODES).
REASON_CORRECT = "correct"
REASON_INCORRECT = "incorrect"
REASON_INVALID_OUTPUT = "invalid_output"


class Outcome(object):
    """A reward value plus the auditable reason that produced it."""

    __slots__ = ("reward", "reason", "detail")

    def __init__(self, reward: float, reason: str, detail: Optional[Dict[str, Any]] = None) -> None:
        if not isinstance(reward, float):
            reward = float(reward)
        if reward != reward:  # NaN guard: a NaN reward poisons the whole group
            raise ValueError("reward must not be NaN")
        self.reward = reward
        self.reason = reason
        self.detail = dict(detail or {})

    @property
    def valid(self) -> bool:
        return self.reason != REASON_INVALID_OUTPUT

    def as_dict(self) -> Dict[str, Any]:
        payload = {"reward": self.reward, "reason": self.reason}
        payload.update(self.detail)
        return payload

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "Outcome(reward={0!r}, reason={1!r})".format(self.reward, self.reason)


def invalid(parse_reason: str, **detail: Any) -> Outcome:
    """An unparsable completion.  Callers decide the numeric value (0 for every method)."""
    payload = {"parse_reason": parse_reason}
    payload.update(detail)
    return Outcome(0.0, REASON_INVALID_OUTPUT, payload)


def scored(correct: bool, detail: Optional[Dict[str, Any]] = None) -> Outcome:
    """A parsable completion worth +1 (correct) or 0 (wrong)."""
    payload = dict(detail or {})
    payload["correct"] = bool(correct)
    return Outcome(1.0 if correct else 0.0,
                   REASON_CORRECT if correct else REASON_INCORRECT,
                   payload)
