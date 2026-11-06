"""Label conventions shared by every experiment.

Absolute label/view invariants enforced here:

* ``answer1_body`` is *response 1* (the HelpSteer3 / RewardBench ``source_chosen``-style slot
  number), ``answer2_body`` is *response 2*.
* A rendered prompt always exposes two **display slots** ``A`` and ``B``.
  ``is_swapped == False``  ->  display A = response 1, display B = response 2.
  ``is_swapped == True``   ->  display A = response 2, display B = response 1.
* ``ground_truth`` is always the letter of the display slot that holds the *better*
  response, i.e. it is expressed **in display space**, never in response space.

The functions below are the only sanctioned way to move between the two spaces, so that an
accidental A/B swap is a type-level (KeyError / ValueError) failure rather than a silent
accuracy bug.
"""

from __future__ import annotations

from typing import Final

ANSWER_A: Final[str] = "A"
ANSWER_B: Final[str] = "B"
VALID_ANSWERS: Final[frozenset[str]] = frozenset({ANSWER_A, ANSWER_B})

# Response identity, in the numbering of the source datasets.
RESPONSE_1: Final[str] = "answer1"
RESPONSE_2: Final[str] = "answer2"
VALID_RESPONSES: Final[frozenset[str]] = frozenset({RESPONSE_1, RESPONSE_2})

_FLIP: Final[dict[str, str]] = {ANSWER_A: ANSWER_B, ANSWER_B: ANSWER_A}

# ``label`` in helpsteer3_train_clean.jsonl is "A" when answer1_body is preferred,
# "B" when answer2_body is preferred.  Verified on the real file: 11,068 rows with
# label=A all have source_overall_preference < 0 and 11,122 rows with label=B all have
# source_overall_preference > 0, which is the HelpSteer3 encoding of "response 2 better".
_SOURCE_LABEL_TO_RESPONSE: Final[dict[str, str]] = {ANSWER_A: RESPONSE_1, ANSWER_B: RESPONSE_2}


def validate_answer(value: str) -> str:
    """Return ``value`` if it is a display-space answer letter, else raise."""
    if value not in VALID_ANSWERS:
        raise ValueError(f"answer must be one of {sorted(VALID_ANSWERS)}, got {value!r}")
    return value


def flip_answer(value: str) -> str:
    """Return the other display slot."""
    return _FLIP[validate_answer(value)]


def source_label_to_response(label: str) -> str:
    """Map a *source-space* ``label`` field onto the response identity it points at."""
    try:
        return _SOURCE_LABEL_TO_RESPONSE[label]
    except KeyError:
        raise ValueError(f"source label must be 'A' or 'B', got {label!r}") from None


def response_to_display(response: str, is_swapped: bool) -> str:
    """Which display slot holds ``response`` for the given ordering."""
    if response not in VALID_RESPONSES:
        raise ValueError(f"response must be one of {sorted(VALID_RESPONSES)}, got {response!r}")
    identity_1_display = ANSWER_B if is_swapped else ANSWER_A
    identity_2_display = ANSWER_A if is_swapped else ANSWER_B
    return identity_1_display if response == RESPONSE_1 else identity_2_display


def display_to_response(display: str, is_swapped: bool) -> str:
    """Inverse of :func:`response_to_display`."""
    validate_answer(display)
    if is_swapped:
        return RESPONSE_2 if display == ANSWER_A else RESPONSE_1
    return RESPONSE_1 if display == ANSWER_A else RESPONSE_2


def ground_truth_for_view(better_response: str, is_swapped: bool) -> str:
    """Display-space ground truth for one of the two equivalent views of a pair."""
    return response_to_display(better_response, is_swapped)


def display_order(is_swapped: bool) -> tuple[str, str]:
    """The (first, second) response identities as laid out in the prompt."""
    return (RESPONSE_2, RESPONSE_1) if is_swapped else (RESPONSE_1, RESPONSE_2)


def is_correct(parsed_display: str | None, ground_truth: str) -> bool:
    """Correctness in display space. ``None`` (unparsable) is always incorrect."""
    validate_answer(ground_truth)
    return parsed_display is not None and parsed_display == ground_truth


def same_response_chosen(
    parsed_original: str | None,
    parsed_swapped: str | None,
    *,
    original_swapped: bool = False,
    swapped_swapped: bool = True,
) -> bool:
    """Position consistency: do the two orderings select the *same physical response*?

    This is deliberately **not** "the two letters agree" -- under a swap the letters must
    differ for a consistent judge.  Returns ``False`` when either view is unparsable.
    """
    if parsed_original is None or parsed_swapped is None:
        return False
    first = display_to_response(parsed_original, original_swapped)
    second = display_to_response(parsed_swapped, swapped_swapped)
    return first == second


def probability_target(ground_truth: str) -> int:
    """The binary target ``y`` for the JwC probability convention.

    ``p = P(display B is better)``; ``y = 1`` iff display B is the better response.
    """
    validate_answer(ground_truth)
    return 1 if ground_truth == ANSWER_B else 0


__all__ = [
    "ANSWER_A",
    "ANSWER_B",
    "RESPONSE_1",
    "RESPONSE_2",
    "VALID_ANSWERS",
    "VALID_RESPONSES",
    "display_order",
    "display_to_response",
    "flip_answer",
    "ground_truth_for_view",
    "is_correct",
    "probability_target",
    "response_to_display",
    "same_response_chosen",
    "source_label_to_response",
    "validate_answer",
]
