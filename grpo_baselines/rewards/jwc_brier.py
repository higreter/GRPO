"""Judging-with-Confidence reward: RL-Brier on an elicited probability.

Paper: arXiv:2510.00263, Sec. 3 Eq. (1)

    R_Brier(tau; y) = 1 - (p - y)^2        if g(tau) = p in [0, 1]
                    = 0                    otherwise (unparsable)

with ``p = Pr(B is better | x)`` and ``y = 1`` when display B holds the better response,
``y = 0`` otherwise.

Three deliberate choices
------------------------
1. **Unparsable -> 0, not a clipped probability.**  Eq. (1) assigns 0 to an unparsable
   completion; giving it ``1 - (0.5 - y)^2 = 0.75`` instead would reward silence.
2. **The 1e-3 epsilon belongs to the log reward only.**  The paper introduces
   ``eps = 1e-3`` to clip the probability inside ``R_Log`` (Eq. 2) so the logarithm stays
   finite.  Brier is quadratic and needs no clip, so :func:`brier_reward` does not clip;
   :func:`log_reward` does, and takes ``eps`` explicitly.
3. **Rewards lie in [0, 1] with a maximum at p == y.**  A perfectly confident correct
   answer gets 1.0, a perfectly confident wrong answer gets 0.0, and the neutral 0.5 gets
   0.75 whatever the label -- which is why the accuracy/calibration pair must be reported
   together, never the reward alone.

This is the *sparse-label* adaptation: the paper's main experiments use a dense
preference distribution (10 annotations per prompt) to define ``p*``; HelpSteer3 provides a
single hard label, so ``y`` is that label and the "true distribution" is degenerate.  The
difference is recorded in the report rather than papered over.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Optional

from common import labels
from common.parsing import ParseResult, get_parser

from .base import Outcome, invalid

#: Paper's clipping constant for the *log* reward (arXiv:2510.00263, Eq. 2).
DEFAULT_EPSILON = 1e-3
#: Neutral probability, used for the "uninformative" reference point in reports.
NEUTRAL_PROBABILITY = 0.5


def brier_reward(probability: float, target: int) -> float:
    """``1 - (p - y)^2`` -- Eq. (1) of the JwC paper."""
    if not math.isfinite(probability):
        raise ValueError("probability must be finite, got {0!r}".format(probability))
    if probability < 0.0 or probability > 1.0:
        raise ValueError("probability must lie in [0, 1], got {0!r}".format(probability))
    if target not in (0, 1):
        raise ValueError("target must be 0 or 1, got {0!r}".format(target))
    return 1.0 - (float(probability) - float(target)) ** 2


def log_reward(probability: float, target: int, *, epsilon: float = DEFAULT_EPSILON) -> float:
    """``y log p' + (1-y) log(1-p')`` with ``p' = clip(p, eps, 1-eps)`` -- Eq. (2).

    Kept for completeness and for the report's comparison table; the shipped configs use
    the Brier reward, which is what the task asked for.
    """
    if not math.isfinite(probability):
        raise ValueError("probability must be finite, got {0!r}".format(probability))
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must lie in [0, 1], got {0!r}".format(probability))
    if target not in (0, 1):
        raise ValueError("target must be 0 or 1, got {0!r}".format(target))
    if not 0.0 < epsilon < 0.5:
        raise ValueError("epsilon must lie in (0, 0.5), got {0!r}".format(epsilon))
    clipped = min(max(float(probability), epsilon), 1.0 - epsilon)
    if target == 1:
        return math.log(clipped)
    return math.log(1.0 - clipped)


def score_parsed(parsed: ParseResult, ground_truth: str) -> Outcome:
    """Reward for one parsed JwC completion."""
    labels.validate_answer(ground_truth)
    target = labels.probability_target(ground_truth)
    if not parsed.ok or parsed.probability is None:
        return invalid(
            parsed.reason,
            target=target,
            ground_truth=ground_truth,
        )
    reward = brier_reward(parsed.probability, target)
    decision = labels.ANSWER_B if parsed.probability > 0.5 else labels.ANSWER_A
    return Outcome(
        reward,
        "scored",
        {
            "probability": parsed.probability,
            "target": target,
            "ground_truth": ground_truth,
            "brier": (parsed.probability - target) ** 2,
            "decision": decision,
            "decision_correct": decision == ground_truth,
            "correct": decision == ground_truth,
        },
    )


def score_text(text: str, ground_truth: str, *, output_format: str = "jwc") -> Outcome:
    return score_parsed(get_parser(output_format)(text), ground_truth)


def make_reward_fn(output_format: str = "jwc") -> Callable[..., list]:
    """TRL adapter: one reward per completion, 0 for unparsable output."""
    parser = get_parser(output_format)

    def reward_fn(
        prompts=None,
        completions=None,
        completion_ids=None,
        ground_truth=None,
        **kwargs: Any,
    ) -> list:
        del prompts, completion_ids, kwargs
        if completions is None or ground_truth is None:
            raise ValueError("completions and ground_truth are required")
        if isinstance(ground_truth, str):
            ground_truth = [ground_truth]
        if len(ground_truth) != len(completions):
            raise ValueError("ground_truth and completions must have equal length")
        rewards = []
        outcomes = []
        for completion, truth in zip(completions, ground_truth):
            text = completion
            if isinstance(text, (list, tuple)) and text and isinstance(text[-1], dict):
                text = text[-1].get("content", "")
            outcome = score_parsed(parser(text or ""), truth)
            outcomes.append(outcome)
            rewards.append(outcome.reward)
        # Sec. 5.4: keep the parsed probability and its reason code alongside the reward.
        reward_fn.last_outcomes = outcomes
        return rewards

    reward_fn.__name__ = "jwc_brier"
    reward_fn.last_outcomes = []
    return reward_fn


def summarise_probabilities(items: list) -> Dict[str, Any]:
    """Mean probability, mean Brier, decision accuracy and invalid rate.

    ``items`` are :class:`rewards.base.Outcome` objects.  Calibration metrics (ECE/MCE)
    live in :mod:`evaluation.metrics` because they need the full label/probability pairs
    from the evaluation JSONL, not a training batch.
    """
    total = len(items)
    if not total:
        raise ValueError("no items to summarise")
    valid = [item for item in items if item.valid]
    probabilities = [item.detail["probability"] for item in valid]
    briers = [item.detail["brier"] for item in valid]
    correct = sum(1 for item in valid if item.detail["decision_correct"])
    return {
        "count": total,
        "valid": len(valid),
        "invalid": total - len(valid),
        "invalid_rate": (total - len(valid)) / total,
        "mean_probability": (sum(probabilities) / len(probabilities)) if probabilities else None,
        "brier": (sum(briers) / len(briers)) if briers else None,
        "decision_accuracy": (correct / len(valid)) if valid else None,
        "mean_reward": sum(item.reward for item in items) / total,
        "neutral_reward_reference": brier_reward(NEUTRAL_PROBABILITY, 1),
    }


__all__ = [
    "DEFAULT_EPSILON",
    "NEUTRAL_PROBABILITY",
    "brier_reward",
    "log_reward",
    "make_reward_fn",
    "score_parsed",
    "score_text",
    "summarise_probabilities",
]
