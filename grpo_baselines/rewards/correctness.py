"""Verdict-correctness reward (vanilla GRPO and the J1 correctness term).

Vanilla GRPO for LLM-as-a-judge: the only signal is "did the judge pick the better
response".  ``+1`` when the parsed verdict matches the display-space ground truth, ``0``
for a wrong verdict **and** ``0`` for an unparsable completion -- but the two zeroes carry
different reason codes so the invalid-output rate stays measurable.

Nothing else contributes to the reward: no length term, no format term, no KL inside the
reward (KL lives in the trainer's loss, as TRL/verl expect).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from common import labels
from common.parsing import ParseResult, get_parser

from .base import Outcome, invalid, scored


def score_verdict(parsed: ParseResult, ground_truth: str) -> Outcome:
    """Reward for a single completion against the display-space ground truth."""
    labels.validate_answer(ground_truth)
    if not parsed.ok or parsed.outcome is None:
        return invalid(parsed.reason, outcome=None, ground_truth=ground_truth)
    correct = labels.is_correct(parsed.outcome, ground_truth)
    return scored(correct, {"outcome": parsed.outcome, "ground_truth": ground_truth})


def score_text(text: str, ground_truth: str, *, output_format: str = "vanilla") -> Outcome:
    """Parse then score -- convenience used by the evaluation path and the tests."""
    parser = get_parser(output_format)
    return score_verdict(parser(text), ground_truth)


def make_reward_fn(output_format: str = "vanilla") -> Callable[..., list]:
    """Build a TRL-compatible reward function.

    TRL calls ``reward_func(prompts=..., completions=..., completion_ids=..., **columns)``
    (trl/trainer/grpo_trainer.py:1648-1706), where ``**columns`` are the remaining dataset
    columns -- in our schema that includes ``ground_truth``.  Returning one float per
    prompt is mandatory, so the length is validated instead of trusting the caller.
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
        if len(ground_truth) != len(completions):
            raise ValueError(
                "ground_truth has {0} entries but there are {1} completions".format(
                    len(ground_truth), len(completions))
            )
        rewards = []
        outcomes = []
        for completion, truth in zip(completions, ground_truth):
            text = completion
            if isinstance(text, (list, tuple)) and text and isinstance(text[-1], dict):
                text = text[-1].get("content", "")
            outcome = score_verdict(parser(text or ""), truth)
            outcomes.append(outcome)
            rewards.append(outcome.reward)
        # Sec. 5.4: the per-rollout log needs the parse result, not just the scalar.
        reward_fn.last_outcomes = outcomes
        return rewards

    reward_fn.__name__ = "verdict_correctness_{0}".format(output_format)
    reward_fn.last_outcomes = []
    return reward_fn


def summary_stats(outcomes: list) -> Dict[str, Any]:
    """Aggregate outcomes for logging: accuracy over parsable items + invalid rate."""
    total = len(outcomes)
    if not total:
        raise ValueError("no outcomes to summarise")
    valid = [item for item in outcomes if item.valid]
    correct = sum(1 for item in valid if item.reward >= 1.0)
    return {
        "count": total,
        "valid": len(valid),
        "invalid": total - len(valid),
        "invalid_rate": (total - len(valid)) / total,
        "accuracy_on_valid": (correct / len(valid)) if valid else None,
        "accuracy_all_treated_wrong": correct / total,
        "mean_reward": sum(item.reward for item in outcomes) / total,
    }


__all__ = [
    "make_reward_fn",
    "score_text",
    "score_verdict",
    "summary_stats",
]
