"""J4R's two-part reward (arXiv:2505.13346v3, Sec. 4, Eq. 6).

.. code-block:: text

    R_j(o, a) = 1      if parse(o) == a           (judgement reward)
                0      otherwise
    R_f(o)    = +0.5   if the output is well formed   (format reward)
                -0.5   otherwise
    R(o, a)   = R_j(o, a) + R_f(o)

Three details are easy to get wrong and are therefore pinned here:

* ``parse(o) == a`` is a *display-space* comparison.  The same physical response is
  ``A`` in one equivalent state and ``B`` in the other, so the ground truth is the
  row's own ``ground_truth`` column, never a pair-level constant.
* A malformed output is **not** a zero.  J4R pays ``-0.5`` for it, which is what
  makes the format term a real signal instead of a tie-breaker.  That is why the
  parser distinguishes "no verdict" (``ok=False``) from "verdict without an
  explanation" (``ok=True, format_ok=False``) -- see :mod:`common.parsing`.
* The format check is *not* re-implemented here.  It reads ``ParseResult.format_ok``,
  so the reward and the evaluation report can never disagree about what "well
  formed" means.

Why this lives under ``trainers/j4r/`` and not ``rewards/``: the reward is
inseparable from the state construction it scores.  The ``+/-0.5`` term is defined
on J4R's ``Explanation:`` / ``Verdict: [[A>B]]`` layout, which is also the layout the
EIS advantage groups over; keeping them in one package makes it obvious that both
belong to the same method.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from common import labels
from common.parsing import ParseResult, get_parser
from rewards.base import (
    REASON_CORRECT,
    REASON_INCORRECT,
    REASON_INVALID_OUTPUT,
    Outcome,
)

#: Eq. 6 constants.  Exposed so the tests can assert the paper's values instead of
#: hard-coding numbers twice.
JUDGEMENT_CORRECT = 1.0
JUDGEMENT_MISS = 0.0
FORMAT_BONUS = 0.5
FORMAT_PENALTY = -0.5

DEFAULT_OUTPUT_FORMAT = "j4r"

#: Every value ``score_parsed`` can return: judgement in {0, 1} plus format in
#: {-0.5, +0.5}.
POSSIBLE_REWARDS = (
    JUDGEMENT_MISS + FORMAT_PENALTY,  # -0.5  wrong/unparsable + malformed
    JUDGEMENT_MISS + FORMAT_BONUS,  # +0.5  wrong + well formed
    JUDGEMENT_CORRECT + FORMAT_PENALTY,  # +0.5  correct + malformed
    JUDGEMENT_CORRECT + FORMAT_BONUS,  # +1.5  correct + well formed
)


def score_parsed(parsed: ParseResult, ground_truth: str) -> Outcome:
    """``R_j + R_f`` for one completion against its own display-space label."""
    labels.validate_answer(ground_truth)
    correct = bool(
        parsed.ok and parsed.outcome is not None and labels.is_correct(parsed.outcome, ground_truth)
    )
    judgement_reward = JUDGEMENT_CORRECT if correct else JUDGEMENT_MISS
    format_reward = FORMAT_BONUS if parsed.format_ok else FORMAT_PENALTY
    total = judgement_reward + format_reward
    detail = {
        "judgement_reward": judgement_reward,
        "format_reward": format_reward,
        "format_ok": bool(parsed.format_ok),
        "outcome": parsed.outcome,
        "ground_truth": ground_truth,
        "correct": correct,
        "parse_reason": parsed.reason,
    }
    if not parsed.ok:
        # Still a scored completion: J4R pays the format penalty, so this is not the
        # "reward 0 for unparsable" convention the other three methods use.
        reason = REASON_INVALID_OUTPUT
    elif correct:
        reason = REASON_CORRECT
    else:
        reason = REASON_INCORRECT
    return Outcome(total, reason, detail)


def score_text(text: str, ground_truth: str, *, output_format: str = DEFAULT_OUTPUT_FORMAT) -> Outcome:
    """Parse then score -- the path shared by the trainer adapter and the report."""
    return score_parsed(get_parser(output_format)(text), ground_truth)


def summarise(outcomes: List[Outcome]) -> Dict[str, Any]:
    """Aggregate a batch so the manifest/log can show both reward terms separately."""
    count = len(outcomes)
    if count == 0:
        return {"count": 0}
    valid = [item for item in outcomes if item.reason != REASON_INVALID_OUTPUT]
    format_ok = [item for item in outcomes if item.detail.get("format_ok")]
    correct = [item for item in outcomes if item.detail.get("correct")]
    rewards = [item.reward for item in outcomes]
    return {
        "count": count,
        "valid": len(valid),
        "invalid": count - len(valid),
        "invalid_rate": (count - len(valid)) / float(count),
        "format_ok": len(format_ok),
        "format_ok_rate": len(format_ok) / float(count),
        "correct_on_valid": len(correct),
        "accuracy_on_valid": (len(correct) / float(len(valid))) if valid else None,
        "accuracy_all_treated_wrong": len(correct) / float(count),
        "mean_judgement_reward": sum(item.detail.get("judgement_reward", 0.0) for item in outcomes)
        / float(count),
        "mean_format_reward": sum(item.detail.get("format_reward", 0.0) for item in outcomes)
        / float(count),
        "mean_reward": sum(rewards) / float(count),
        "min_reward": min(rewards),
        "max_reward": max(rewards),
    }


def make_reward_fn(output_format: str = DEFAULT_OUTPUT_FORMAT) -> Callable[..., list]:
    """Build the TRL-compatible reward function.

    TRL calls ``reward_fn(prompts=..., completions=..., completion_ids=..., **columns)``
    (``trl/trainer/grpo_trainer.py:1648-1706``) and requires exactly one float per
    completion.  The per-row values are also stashed on ``last_rewards`` because
    :class:`trainers.j4r.advantage.EISAdvantageMixin` needs the *same* numbers TRL
    scored with -- recomputing them from decoded text would be a second, silently
    different parse.
    """
    parser = get_parser(output_format)

    def reward_fn(
        prompts=None,
        completions=None,
        completion_ids=None,
        ground_truth=None,
        **kwargs: Any,
    ) -> list:
        del prompts, completion_ids, kwargs
        if completions is None:
            raise ValueError("completions are required")
        if ground_truth is None:
            raise ValueError("ground_truth column is required (display-space A/B letters)")
        if isinstance(ground_truth, str):
            ground_truth = [ground_truth]
        if len(completions) != len(ground_truth):
            raise ValueError(
                "completions ({0}) and ground_truth ({1}) must have equal length".format(
                    len(completions), len(ground_truth))
            )
        outcomes = [
            score_parsed(parser(_completion_text(completion)), truth)
            for completion, truth in zip(completions, ground_truth)
        ]
        values = [item.reward for item in outcomes]
        reward_fn.last_outcomes = outcomes
        reward_fn.last_rewards = values
        reward_fn.last_stats = summarise(outcomes)
        return values

    reward_fn.last_outcomes = []
    reward_fn.last_rewards = []
    reward_fn.last_stats = {}
    reward_fn.__name__ = "j4r_judgement_and_format"
    return reward_fn


def _completion_text(completion: Any) -> str:
    """Accept either a raw string or TRL's conversational completion list."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, (list, tuple)) and completion:
        last = completion[-1]
        if isinstance(last, dict):
            return last.get("content", "")
    return str(completion)


__all__ = [
    "DEFAULT_OUTPUT_FORMAT",
    "FORMAT_BONUS",
    "FORMAT_PENALTY",
    "JUDGEMENT_CORRECT",
    "JUDGEMENT_MISS",
    "POSSIBLE_REWARDS",
    "make_reward_fn",
    "score_parsed",
    "score_text",
    "summarise",
]
