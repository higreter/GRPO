"""Per-rollout records and per-group health diagnostics.

Two prompt requirements land in the same place, so they share a module:

* Sec. 5.4 -- a TRL-compatible multi-rollout group interface that keeps, for every
  rollout, its prompt identity, its completion, its parse result and its reward.
* Sec. 5.5 -- explicit handling and recording of the degenerate cases: every reward in
  a group identical, standard deviation zero, or too few valid rollouts to normalise.

TRL computes the advantage with

    advantages = rewards - mean_grouped_rewards
    advantages = advantages / (std_rewards + 1e-4)

and then ``torch.nan_to_num(advantages, nan=0.0)``. That is numerically safe but it is
*silent*: a group where all eight rollouts produced the same reward yields an advantage
of exactly zero for every rollout, which carries no learning signal, and nothing in the
loss says so. If a parser breaks and every completion becomes unparsable, every group in
the batch is flat and the run looks like a healthy zero-loss plateau. These diagnostics
exist so that case is visible in the log rather than inferred from a flat curve later.

The module is pure standard library: it runs on CPU, in tests, before any training stack
exists.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from rewards.base import REASON_INVALID_OUTPUT

# A group needs at least this many usable rollouts before its normalisation means
# anything. GRPO's own floor is 2 (a group of one has no spread); the config layer
# enforces that separately and this constant keeps the diagnostics agreeing with it.
MIN_VALID_ROLLOUTS = 2

# Below this a group is treated as having no spread. Matches the epsilon TRL adds to the
# standard deviation in `advantages / (std_rewards + 1e-4)`.
ZERO_VARIANCE_EPSILON = 1e-4


class RolloutLogError(ValueError):
    """Raised when a rollout log cannot be built from the batch it was given."""


class RolloutRecord(object):
    """One generated completion, with everything needed to audit its reward.

    ``prompt_id`` is the dataset's ``sample_id`` (``"<pair_id>::<view>"``), which is what
    ties a rollout back to its source pair and to its counterpart state. The remaining
    fields mirror what the brief asks to preserve: completion, parse result, reward.
    """

    __slots__ = ("prompt_id", "pair_id", "view", "rollout_index", "completion",
                 "reward", "reason", "outcome", "format_ok", "group_key", "group_id")

    def __init__(self, prompt_id, completion, reward, *, pair_id=None, view=None,
                 rollout_index=None, reason=None, outcome=None, format_ok=None,
                 group_key=None, group_id=None):
        if not isinstance(prompt_id, str) or not prompt_id:
            raise RolloutLogError(
                "prompt_id must be a non-empty string, got {0!r}".format(prompt_id))
        if reward is not None:
            reward = float(reward)
            if math.isnan(reward):
                raise RolloutLogError(
                    "rollout {0!r} has a NaN reward; a NaN would poison its whole "
                    "group's mean and standard deviation".format(prompt_id))
        self.prompt_id = prompt_id
        self.completion = completion
        self.reward = reward
        self.pair_id = pair_id
        self.view = view
        self.rollout_index = rollout_index
        self.reason = reason
        self.outcome = outcome
        self.format_ok = format_ok
        # The normalisation group. GRPO groups by prompt; J4R's EIS groups by state
        # inside a pair. Defaulting to the prompt id keeps vanilla and J1 honest.
        self.group_key = group_key if group_key is not None else prompt_id
        # Source-dataset group (brief Sec. 3.3). Optional: a hand-built record from a test
        # or a replay need not come from HelpSteer at all, and inventing a value would be
        # worse than leaving it empty.
        self.group_id = group_id

    @property
    def valid(self) -> bool:
        """Usable means the completion parsed.

        An unparsable completion still receives a reward -- 0.0 for vanilla/J1/JwC, and
        -0.5 for J4R's format penalty -- so "has a reward" cannot be the test; the
        reason code can.  When no reason was recorded the reward's presence is the only
        evidence available, and it is used rather than guessed.
        """
        if self.reason is not None:
            return self.reason != REASON_INVALID_OUTPUT
        return self.reward is not None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "prompt_id": self.prompt_id,
            "pair_id": self.pair_id,
            "view": self.view,
            "rollout_index": self.rollout_index,
            "completion": self.completion,
            "reward": self.reward,
            "reason": self.reason,
            "outcome": self.outcome,
            "format_ok": self.format_ok,
            "group_key": self.group_key,
            "group_id": self.group_id,
        }


def _stats(values: Sequence[float]) -> Dict[str, float]:
    """Mean and population standard deviation of a non-empty sequence."""
    count = len(values)
    mean = sum(values) / count
    variance = sum((value - mean) ** 2 for value in values) / count
    return {"count": count, "mean": mean, "std": math.sqrt(variance),
            "min": min(values), "max": max(values)}


def group_diagnostics(records: Iterable[RolloutRecord], *,
                      min_valid_rollouts: int = MIN_VALID_ROLLOUTS,
                      epsilon: float = ZERO_VARIANCE_EPSILON) -> Dict[str, Any]:
    """Summarise every normalisation group, calling out the degenerate ones.

    Returns a ``groups`` mapping plus named problem lists. ``flat_groups`` and
    ``underfilled_groups`` are *reported*, not raised: a flat group legitimately
    produces zero advantage and training should continue. What must not happen is that
    it goes unnoticed.
    """
    records = list(records)
    if not records:
        raise RolloutLogError("no rollouts to diagnose")

    by_group: Dict[str, List[RolloutRecord]] = {}
    for record in records:
        by_group.setdefault(record.group_key, []).append(record)

    groups: Dict[str, Any] = {}
    flat: List[str] = []
    underfilled: List[str] = []
    incomplete: List[str] = []

    for key in sorted(by_group):
        members = by_group[key]
        # Every reward in the group, invalid parses included: these are exactly the
        # numbers TRL puts through mean/std, so the flat test must use them too.
        rewards = [record.reward for record in members if record.reward is not None]
        valid_count = sum(1 for record in members if record.valid)
        entry: Dict[str, Any] = {
            "rollouts": len(members),
            "valid": valid_count,
            "invalid": len(members) - valid_count,
            "scored": len(rewards),
            "rollout_indices": [record.rollout_index for record in members],
        }
        if len(rewards) < 2:
            # Fewer than two usable rewards: no spread is even definable. Report the
            # mean of what exists so a single-rollout group is still legible, but mark
            # the group as carrying no signal rather than inventing a standard
            # deviation of zero.
            entry["stats"] = ({"count": len(rewards),
                               "mean": rewards[0] if rewards else None,
                               "std": None, "min": rewards[0] if rewards else None,
                               "max": rewards[0] if rewards else None})
            entry["flat"] = True
            entry["zero_variance"] = True
            entry["advantage_signal"] = False
        else:
            stats = _stats(rewards)
            entry["stats"] = stats
            # All-equal and zero-std are the same condition here; both keys are
            # reported because the brief names both and a reader may look for either.
            entry["flat"] = entry["zero_variance"] = stats["std"] < epsilon
            entry["advantage_signal"] = not entry["flat"]
        # Underfilled is about measurement (how many rollouts actually parsed); flat is
        # about arithmetic (did TRL see any spread). A group of eight 0.0s from broken
        # parses is both.
        if valid_count < min_valid_rollouts:
            underfilled.append(key)
        elif entry["flat"]:
            flat.append(key)
        if valid_count < len(members):
            incomplete.append(key)
        groups[key] = entry

    flat_rollouts = sum(groups[key]["rollouts"] for key in flat)
    total = len(records)
    return {
        "groups": groups,
        "group_count": len(groups),
        "rollouts": total,
        "valid": sum(1 for record in records if record.valid),
        "invalid": sum(1 for record in records if not record.valid),
        "flat_groups": flat,
        "flat_group_count": len(flat),
        "flat_rollout_count": flat_rollouts,
        "flat_rollout_ratio": flat_rollouts / total,
        "underfilled_groups": underfilled,
        "underfilled_group_count": len(underfilled),
        "incomplete_groups": incomplete,
        "incomplete_group_count": len(incomplete),
        "epsilon": epsilon,
        "min_valid_rollouts": min_valid_rollouts,
        "note": (
            "flat groups yield an advantage of zero for every rollout: correct "
            "arithmetic, no learning signal. A batch whose flat_rollout_ratio is near 1 "
            "usually means the parser or the reward is broken, not that the policy has "
            "converged."
        ),
    }


def split_sample_id(sample_id: str):
    """``"<pair_id>::<view>"`` -> ``(pair_id, view)``; ``view`` is None if absent."""
    pair_id, separator, view = sample_id.partition("::")
    return pair_id, (view if separator else None)


def records_from_batch(completions, rewards, sample_ids, *, reasons=None,
                       outcomes=None, formats=None, group_key_fn=None,
                       group_ids=None) -> List[RolloutRecord]:
    """Build one record per rollout from the parallel lists TRL hands a reward fn.

    ``group_key_fn`` defaults to the prompt id. J4R overrides it so a group is one state
    inside a pair, which is what EIS normalises within.
    """
    completions = list(completions)
    sample_ids = list(sample_ids)
    rewards = list(rewards)
    if not (len(completions) == len(sample_ids) == len(rewards)):
        raise RolloutLogError(
            "completions ({0}), sample_ids ({1}) and rewards ({2}) must have equal "
            "length".format(len(completions), len(sample_ids), len(rewards)))

    out_reasons = list(reasons) if reasons is not None else [None] * len(completions)
    out_outcomes = list(outcomes) if outcomes is not None else [None] * len(completions)
    out_formats = list(formats) if formats is not None else [None] * len(completions)

    if group_ids is None:
        group_ids = [None] * len(completions)
    else:
        group_ids = list(group_ids)
        if len(group_ids) != len(completions):
            raise RolloutLogError(
                "group_ids has {0} entries but there are {1} rollouts".format(
                    len(group_ids), len(completions)))

    counters: Dict[str, int] = {}
    records: List[RolloutRecord] = []
    for completion, reward, sample_id, reason, outcome, format_ok, group_id in zip(
            completions, rewards, sample_ids, out_reasons, out_outcomes, out_formats,
            group_ids):
        pair_id, view = split_sample_id(sample_id)
        group_key = group_key_fn(sample_id) if group_key_fn is not None else sample_id
        index = counters.get(group_key, 0)
        counters[group_key] = index + 1
        records.append(RolloutRecord(
            sample_id, completion, reward, pair_id=pair_id, view=view,
            rollout_index=index, reason=reason, outcome=outcome,
            format_ok=format_ok, group_key=group_key, group_id=group_id))
    return records


def _outcome_details(outcomes, count):
    """Pull (reason, format_ok) out of a reward function's recorded outcomes.

    Not every reward function keeps its outcomes, and those that do keep objects of
    different shapes (``rewards.base.Outcome`` here, something else if a caller swaps in
    its own). Anything unexpected degrades to ``None`` rather than guessing: a wrong
    reason code in a log is worse than a missing one.
    """
    if outcomes is None:
        return [None] * count, [None] * count
    outcomes = list(outcomes)
    if len(outcomes) != count:
        return [None] * count, [None] * count
    reasons: List[Optional[str]] = []
    formats: List[Optional[bool]] = []
    for item in outcomes:
        reasons.append(getattr(item, "reason", None))
        detail = getattr(item, "detail", None)
        formats.append(detail.get("format_ok") if isinstance(detail, dict) else None)
    return reasons, formats


def logged_reward_fn(reward_fn: Callable, *, group_key_fn=None) -> Callable:
    """Wrap a TRL reward function so each call also records its rollouts.

    The wrapper is transparent: it returns exactly what ``reward_fn`` returned. The
    records land on ``wrapper.last_rollouts`` and the diagnostics on
    ``wrapper.last_diagnostics``, which is how a trainer logs Sec. 5.4's per-rollout
    detail and Sec. 5.5's flat-group warning without changing TRL's contract.
    """
    def wrapper(prompts=None, completions=None, completion_ids=None,
                sample_id=None, ground_truth=None, group_id=None, **kwargs):
        rewards = reward_fn(prompts=prompts, completions=completions,
                            completion_ids=completion_ids, sample_id=sample_id,
                            ground_truth=ground_truth, **kwargs)
        rewards = list(rewards)

        ids = [""] * len(rewards) if sample_id is None else list(sample_id)
        if len(ids) != len(rewards):
            raise RolloutLogError(
                "sample_id has {0} entries but the reward function returned {1}"
                .format(len(ids), len(rewards)))

        reasons, formats = _outcome_details(
            getattr(reward_fn, "last_outcomes", None), len(rewards))

        wrapper.last_rollouts = records_from_batch(
            completions if completions is not None else [None] * len(rewards),
            rewards, ids, reasons=reasons, formats=formats, group_key_fn=group_key_fn,
            group_ids=list(group_id) if group_id is not None else None)
        wrapper.last_diagnostics = group_diagnostics(wrapper.last_rollouts)
        return rewards

    wrapper.__name__ = getattr(reward_fn, "__name__", "logged_reward")
    wrapper.__doc__ = getattr(reward_fn, "__doc__", None)
    wrapper.last_rollouts = []
    wrapper.last_diagnostics = None
    wrapper.wrapped = reward_fn
    return wrapper


__all__ = [
    "MIN_VALID_ROLLOUTS",
    "RolloutLogError",
    "RolloutRecord",
    "ZERO_VARIANCE_EPSILON",
    "group_diagnostics",
    "logged_reward_fn",
    "records_from_batch",
    "split_sample_id",
]
