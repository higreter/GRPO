"""Construction of linked prompt *states* for one source preference pair.

This module is the single place where "which physical answer sits in display slot A"
is decided.  Everything downstream -- prompt rendering, rewards, evaluation, logging --
consumes the :class:`PairView` objects produced here and never re-derives the mapping.

Two naming spaces appear everywhere in the project and must not be confused:

``response`` space (physical identity)
    ``"answer1"`` / ``"answer2"``: the two responses as stored in the dataset.  Immutable
    under reordering.
``display`` space (what the judge sees)
    ``"A"`` / ``"B"``: the first and second slot of the prompt.  A swap changes this
    mapping without changing any content.

A :class:`PairView` carries both, plus the ground-truth *display* letter, so a label can
never be reused across a swap by accident: the constructor recomputes
``ground_truth`` from ``better_response`` and ``is_swapped`` and verifies the caller did
not smuggle in a stale one.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional

from common import labels

SOURCE_ORIGINAL = "original"
SOURCE_SWAPPED = "swapped"
VIEW_NAMES = (SOURCE_ORIGINAL, SOURCE_SWAPPED)


def _stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha1("\x1f".join(parts).encode("utf-8")).hexdigest()[:16]
    return "{0}_{1}".format(prefix, digest)


class PairView(object):
    """One prompt state: a fixed display order plus the ground truth for that order."""

    __slots__ = (
        "sample_id",
        "pair_id",
        "source_row_index",
        "question",
        "answer_a",
        "answer_b",
        "better_response",
        "is_swapped",
        "ground_truth",
        "source_dataset",
        "view",
        "metadata",
        "source_line",
        "group_id",
    )

    def __init__(
        self,
        pair_id: str,
        source_row_index: int,
        question: str,
        answer_1: str,
        answer_2: str,
        better_response: str,
        is_swapped: bool,
        source_dataset: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not isinstance(pair_id, str) or not pair_id:
            raise ValueError("pair_id must be a non-empty str")
        if better_response not in (labels.RESPONSE_1, labels.RESPONSE_2):
            raise ValueError(
                "better_response must be one of {0!r}/{1!r}, got {2!r}".format(
                    labels.RESPONSE_1, labels.RESPONSE_2, better_response)
            )
        if not isinstance(is_swapped, bool):
            raise TypeError("is_swapped must be a bool, got {0!r}".format(type(is_swapped)))
        for name, value in (("question", question), ("answer_1", answer_1),
                            ("answer_2", answer_2)):
            if not isinstance(value, str):
                raise TypeError("{0} must be a str".format(name))
            if not value.strip():
                raise ValueError("{0} must not be empty".format(name))

        first, second = labels.display_order(is_swapped)
        body = {labels.RESPONSE_1: answer_1, labels.RESPONSE_2: answer_2}
        self.pair_id = pair_id
        self.source_row_index = int(source_row_index)
        self.question = question
        self.answer_a = body[first]
        self.answer_b = body[second]
        self.better_response = better_response
        self.is_swapped = is_swapped
        self.ground_truth = labels.ground_truth_for_view(better_response, is_swapped)
        self.source_dataset = source_dataset
        self.view = SOURCE_SWAPPED if is_swapped else SOURCE_ORIGINAL
        self.metadata = dict(metadata or {})
        # Traceability (brief Sec. 3.3): source_line and group_id travel with the sample
        # as ordinary record fields, not only inside `metadata`.  `to_record()` is what
        # feeds every adapter output, TRL row and rollout log, and it used to drop both,
        # leaving `pair_id` string-surgery as the only route back to the source group.
        self.source_line = self.metadata.get("source_line")
        self.group_id = self.metadata.get("group_id")
        # sample_id must be unique per (pair, state) and stable across runs.
        self.sample_id = "{0}::{1}".format(pair_id, self.view)

    # -- serialisation -----------------------------------------------------------
    def to_record(self) -> Dict[str, Any]:
        """The minimal schema every dataset row and log line carries."""
        return {
            "sample_id": self.sample_id,
            "pair_id": self.pair_id,
            "question": self.question,
            "answer_a": self.answer_a,
            "answer_b": self.answer_b,
            "ground_truth": self.ground_truth,
            "is_swapped": self.is_swapped,
            "source_dataset": self.source_dataset,
            "view": self.view,
            "better_response": self.better_response,
            "source_row_index": self.source_row_index,
            "source_line": self.source_line,
            "group_id": self.group_id,
        }

    def as_dict(self) -> Dict[str, Any]:
        payload = self.to_record()
        payload["metadata"] = dict(self.metadata)
        return payload

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "PairView(pair_id={0!r}, view={1!r}, gt={2!r})".format(
            self.pair_id, self.view, self.ground_truth)


def build_views(
    pair_id: str,
    source_row_index: int,
    question: str,
    answer_1: str,
    answer_2: str,
    better_response: str,
    source_dataset: str,
    *,
    two_states: bool,
    metadata: Optional[Dict[str, Any]] = None,
) -> List[PairView]:
    """Build the prompt state(s) for one source pair.

    ``two_states=True`` returns ``[original, swapped]`` -- the ``T_1(q) .. T_L(q)``
    construction used by J1 (arXiv:2505.10320 Sec. 2.2 consistency reward) and J4R
    (arXiv:2505.13346 Sec. 3.2 EIS).  ``two_states=False`` returns the original only,
    used by vanilla GRPO and by JwC, whose Brier target ``y`` is defined on a single
    fixed orientation.
    """
    original = PairView(
        pair_id=pair_id,
        source_row_index=source_row_index,
        question=question,
        answer_1=answer_1,
        answer_2=answer_2,
        better_response=better_response,
        is_swapped=False,
        source_dataset=source_dataset,
        metadata=metadata,
    )
    if not two_states:
        return [original]
    swapped = PairView(
        pair_id=pair_id,
        source_row_index=source_row_index,
        question=question,
        answer_1=answer_1,
        answer_2=answer_2,
        better_response=better_response,
        is_swapped=True,
        source_dataset=source_dataset,
        metadata=metadata,
    )
    # Guard rails: the two states must reference identical content and opposite letters.
    if swapped.answer_a != original.answer_b or swapped.answer_b != original.answer_a:
        raise AssertionError("swapped view did not exchange the two responses")
    if swapped.ground_truth == original.ground_truth:
        raise AssertionError(
            "swapped view kept ground_truth={0!r}; the label must flip".format(
                swapped.ground_truth)
        )
    return [original, swapped]


def group_by_pair(views: List[PairView]) -> Dict[str, List[PairView]]:
    """Group views by ``pair_id`` preserving order; validates the grouping."""
    grouped: Dict[str, List[PairView]] = {}
    for view in views:
        grouped.setdefault(view.pair_id, []).append(view)
    for pair_id, members in grouped.items():
        views_seen = [member.view for member in members]
        if len(set(views_seen)) != len(views_seen):
            raise ValueError("duplicate view for pair {0!r}: {1}".format(pair_id, views_seen))
        for member in members:
            if member.pair_id != pair_id:
                raise AssertionError("grouping produced a foreign member")
    return grouped


def identity_of(view: PairView, display_letter: Optional[str]) -> Optional[str]:
    """Map a display letter of ``view`` back to the physical response identity."""
    if display_letter is None:
        return None
    if display_letter not in (labels.ANSWER_A, labels.ANSWER_B):
        raise ValueError("display_letter must be A/B/None, got {0!r}".format(display_letter))
    return labels.display_to_response(display_letter, view.is_swapped)


def same_choice_across_views(original: PairView, swapped: PairView,
                             original_letter: Optional[str],
                             swapped_letter: Optional[str]) -> bool:
    """True when both states pick the *same physical response*.

    This is the J1 consistency criterion (arXiv:2505.10320 Sec. 2.2) expressed on
    identities rather than letters: because the swap flips the letters, comparing the two
    letters directly would report "consistent" exactly when the judge flipped its answer.
    """
    if not original.is_swapped and swapped.is_swapped:
        pass
    elif original.is_swapped and not swapped.is_swapped:
        original, swapped = swapped, original
        original_letter, swapped_letter = swapped_letter, original_letter
    else:
        raise ValueError("expected one original and one swapped view")
    if original.pair_id != swapped.pair_id:
        raise ValueError("views belong to different pairs")
    first = identity_of(original, original_letter)
    second = identity_of(swapped, swapped_letter)
    if first is None or second is None:
        return False
    return first == second


def assert_views_cover(views: List[PairView], expected_pairs: int) -> None:
    """Validate that ``views`` covers ``expected_pairs`` distinct pairs consistently."""
    grouped = group_by_pair(views)
    if len(grouped) != expected_pairs:
        raise ValueError(
            "expected {0} pairs, got {1}".format(expected_pairs, len(grouped))
        )
    for pair_id, members in grouped.items():
        identities = {member.better_response for member in members}
        if len(identities) != 1:
            raise ValueError(
                "pair {0!r} disagrees about the better response: {1}".format(
                    pair_id, sorted(identities))
            )
        orders = {tuple(sorted([m.answer_a, m.answer_b])) for m in members}
        if len(orders) != 1:
            raise ValueError("pair {0!r} has inconsistent response content".format(pair_id))


__all__ = [
    "PairView",
    "SOURCE_ORIGINAL",
    "SOURCE_SWAPPED",
    "VIEW_NAMES",
    "assert_views_cover",
    "build_views",
    "group_by_pair",
    "identity_of",
    "same_choice_across_views",
]
