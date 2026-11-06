"""J1 reward: verdict correctness + verdict consistency over the two linked orders.

The J1 paper (arXiv:2505.10320, Sec. 2.2) defines exactly two reward terms and nothing
else:

``Verdict Correctness``
    ``+1`` when the final verdict is right, ``0`` otherwise (unparsable counts as wrong).
``Verdict Consistency``
    ``+1`` **only when both orders of the same pair are judged correctly**, ``0`` if either
    one is wrong or unparsable.

Total reward for one pair is therefore ``correct(o1) + correct(o2) +
both_correct(o1, o2)``, i.e. a pair can earn 0, 1, 2 or 3.

Why this cannot be reduced to "the two letters agree": the swapped state flips the
ground-truth letter, so a judge that ignores position produces *different* letters and is
correct, while a position-biased judge produces the *same* letter and is wrong.  The
consistency term is computed on physical response identity through
:func:`common.labels.same_response_chosen`.

TRL note: GRPO normalises advantages *within* a group.  To make the pair-level term
meaningful the consistency reward is broadcast to both rows of the pair (each row's reward
contains the same consistency bonus), which keeps the group mean interpretable and lets the
trainer stay unmodified.  The consequence is documented in the report: J1's effective
per-completion reward is ``correct + both_correct``.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence

from common import labels
from common.parsing import ParseResult, get_parser

from .base import Outcome, invalid

CONSISTENCY_BONUS = 1.0
CORRECTNESS_BONUS = 1.0


def score_pair(
    original_parsed: ParseResult,
    swapped_parsed: ParseResult,
    original_ground_truth: str,
    swapped_ground_truth: str,
) -> Dict[str, Any]:
    """Score one (original, swapped) pair and return both rows plus diagnostics."""
    labels.validate_answer(original_ground_truth)
    labels.validate_answer(swapped_ground_truth)
    if swapped_ground_truth == original_ground_truth:
        raise ValueError(
            "origin and swapped views must have opposite ground truths, both are {0!r}".format(
                original_ground_truth)
        )

    original_correct = labels.is_correct(original_parsed.outcome, original_ground_truth) \
        if original_parsed.ok else False
    swapped_correct = labels.is_correct(swapped_parsed.outcome, swapped_ground_truth) \
        if swapped_parsed.ok else False

    # Consistency is the paper's both-order term: both verdicts must be right.
    both_correct = bool(original_correct and swapped_correct)

    # Cross-check on identities: for correct verdicts this must agree with both_correct.
    same_identity = labels.same_response_chosen(
        original_parsed.outcome, swapped_parsed.outcome,
        original_swapped=False, swapped_swapped=True,
    )

    original_reward = (CORRECTNESS_BONUS if original_correct else 0.0) + (
        CONSISTENCY_BONUS if both_correct else 0.0)
    swapped_reward = (CORRECTNESS_BONUS if swapped_correct else 0.0) + (
        CONSISTENCY_BONUS if both_correct else 0.0)

    return {
        "original_reward": original_reward,
        "swapped_reward": swapped_reward,
        "original_correct": original_correct,
        "swapped_correct": swapped_correct,
        "both_correct": both_correct,
        "same_response_chosen": same_identity,
        "original_outcome": original_parsed.outcome,
        "swapped_outcome": swapped_parsed.outcome,
        "original_reason": original_parsed.reason,
        "swapped_reason": swapped_parsed.reason,
        "pair_reward": original_reward + swapped_reward,
    }


def score_texts(
    original_text: str,
    swapped_text: str,
    original_ground_truth: str,
    swapped_ground_truth: str,
    *,
    output_format: str = "j1",
) -> Dict[str, Any]:
    parser = get_parser(output_format)
    return score_pair(
        parser(original_text), parser(swapped_text),
        original_ground_truth, swapped_ground_truth,
    )


def make_reward_fn(output_format: str = "j1") -> Callable[..., list]:
    """TRL adapter that receives both rows of a pair in one batch.

    ``sample_id`` encodes ``"<pair_id>::<view>"`` (see :class:`data.pair_views.PairView`),
    so the function can regroup the batch by pair even after shuffling.  Rows whose
    counterpart is missing from the batch keep their correctness term and receive no
    consistency bonus; that count is reported through ``stats`` so a silently broken
    sampler cannot hide.
    """
    parser = get_parser(output_format)

    def reward_fn(
        prompts=None,
        completions=None,
        completion_ids=None,
        ground_truth=None,
        sample_id=None,
        **kwargs: Any,
    ) -> list:
        del prompts, completion_ids, kwargs
        if completions is None or ground_truth is None:
            raise ValueError("completions and ground_truth are required")
        if sample_id is None:
            raise ValueError(
                "sample_id is required: J1 pairs rows through '<pair_id>::<view>'"
            )
        if not (len(completions) == len(ground_truth) == len(sample_id)):
            raise ValueError("completions/ground_truth/sample_id must have equal length")

        parsed = [parser(_completion_text(item)) for item in completions]
        rows: Dict[str, Dict[str, Any]] = {}
        for index, sid in enumerate(sample_id):
            pair_id, _, view = str(sid).partition("::")
            if not view:
                raise ValueError("sample_id {0!r} is not '<pair_id>::<view>'".format(sid))
            members = rows.setdefault(pair_id, {})
            if view in members:
                # Overwriting would leave the earlier row at its initial 0.0 without any
                # reward being computed for it -- a silent wrong signal.  The sampler
                # guarantees one row per (pair, view), so a duplicate means the batch or
                # the dataset is malformed.
                raise ValueError(
                    "duplicate view {0!r} for pair {1!r} in one generation batch "
                    "(rows {2} and {3}); every (pair, view) must appear exactly once".format(
                        view, pair_id, members[view], index)
                )
            members[view] = index

        rewards = [0.0] * len(completions)
        unpaired = 0
        paired = 0
        for pair_id, members in rows.items():
            if "original" in members and "swapped" in members:
                i, j = members["original"], members["swapped"]
                result = score_pair(parsed[i], parsed[j], ground_truth[i], ground_truth[j])
                rewards[i] = result["original_reward"]
                rewards[j] = result["swapped_reward"]
                paired += 1
            else:
                unpaired += 1
                for index in members.values():
                    correct = labels.is_correct(parsed[index].outcome, ground_truth[index]) \
                        if parsed[index].ok else False
                    rewards[index] = CORRECTNESS_BONUS if correct else 0.0
        reward_fn.last_stats = {
            "paired_groups": paired,
            "unpaired_groups": unpaired,
            "rows": len(completions),
        }
        # Sec. 5.4: keep the parse result per row, so the rollout log can say whether a
        # zero came from a wrong verdict or from an unparsable completion.
        reward_fn.last_outcomes = parsed
        return rewards

    reward_fn.last_stats = {}
    reward_fn.last_outcomes = []
    reward_fn.__name__ = "j1_verdict_and_consistency"
    return reward_fn


def _completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, (list, tuple)) and completion:
        last = completion[-1]
        if isinstance(last, dict):
            return last.get("content", "")
    return str(completion)


__all__ = [
    "CONSISTENCY_BONUS",
    "CORRECTNESS_BONUS",
    "make_reward_fn",
    "score_pair",
    "score_texts",
]
