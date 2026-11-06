"""Output parsers for the four baseline methods.

Every parser is registered under an output-format name so that a config's
``output_format`` selects it unambiguously; nothing downstream sniffs the text with
ad-hoc heuristics.

Returned :class:`ParseResult` deliberately separates three questions that the reward
functions must not conflate:

``ok``
    Did we recover a usable decision (a letter, or a probability)?
``format_ok``
    Does the completion structurally match the layout the prompt asked for?  Only J4R
    turns this into a reward term (Sec. 4 of arXiv:2505.13346: ``+0.5`` well formatted,
    ``-0.5`` otherwise); for the other methods it is recorded for diagnostics because
    J1's paper explicitly chose *not* to add a format reward.
``reason``
    A stable machine-readable code from :data:`REASON_CODES`, which is what the
    evaluation JSONL and the reward logs record instead of free text.

Pseudo-matches are rejected on purpose: ``"the answer is A"`` without the required tag
never parses, and an inner payload such as ``[[A>B]]`` is reported as an invalid letter
rather than silently truncated to ``A``.
"""

from __future__ import annotations

import math
import re
from typing import Dict, Optional

# --------------------------------------------------------------------------------------
# Reason codes.  Frozen so that logs and tests can assert on them.
# --------------------------------------------------------------------------------------
OK = "ok"
EMPTY_OUTPUT = "empty_output"
NO_ANSWER_TAG = "no_answer_tag"
UNCLOSED_ANSWER_TAG = "unclosed_answer_tag"
MULTIPLE_ANSWER_TAGS = "multiple_answer_tags"
INVALID_ANSWER_LETTER = "invalid_answer_letter"
MISSING_THINK = "missing_think"
UNCLOSED_THINK = "unclosed_think"
MULTIPLE_THINK_BLOCKS = "multiple_think_blocks"
EMPTY_THINK = "empty_think"
THINK_AFTER_ANSWER = "think_after_answer"
ANSWER_NOT_BRACKETED = "answer_not_bracketed"
NO_VERDICT_KEYWORD = "no_verdict_keyword"
MULTIPLE_VERDICT_KEYWORDS = "multiple_verdict_keywords"
NO_VERDICT_TAG = "no_verdict_tag"
CONFLICTING_VERDICTS = "conflicting_verdicts"
MISSING_EXPLANATION = "missing_explanation"
NO_PROB_SECTION = "no_prob_section"
MULTIPLE_PROB_TAGS = "multiple_prob_tags"
UNCLOSED_PROB_TAG = "unclosed_prob_tag"
PROB_NOT_A_NUMBER = "prob_not_a_number"
PROB_NOT_FINITE = "prob_not_finite"
PROB_OUT_OF_RANGE = "prob_out_of_range"
UNKNOWN_FORMAT = "unknown_format"

REASON_CODES = (
    OK,
    EMPTY_OUTPUT,
    NO_ANSWER_TAG,
    UNCLOSED_ANSWER_TAG,
    MULTIPLE_ANSWER_TAGS,
    INVALID_ANSWER_LETTER,
    MISSING_THINK,
    UNCLOSED_THINK,
    MULTIPLE_THINK_BLOCKS,
    EMPTY_THINK,
    THINK_AFTER_ANSWER,
    ANSWER_NOT_BRACKETED,
    NO_VERDICT_KEYWORD,
    MULTIPLE_VERDICT_KEYWORDS,
    NO_VERDICT_TAG,
    CONFLICTING_VERDICTS,
    MISSING_EXPLANATION,
    NO_PROB_SECTION,
    MULTIPLE_PROB_TAGS,
    UNCLOSED_PROB_TAG,
    PROB_NOT_A_NUMBER,
    PROB_NOT_FINITE,
    PROB_OUT_OF_RANGE,
    UNKNOWN_FORMAT,
)


# --------------------------------------------------------------------------------------
# Result container
# --------------------------------------------------------------------------------------
class ParseResult(object):
    """Immutable-by-convention parse outcome."""

    __slots__ = ("ok", "format_ok", "outcome", "probability", "reasoning", "reason", "raw")

    def __init__(
        self,
        ok: bool,
        reason: str,
        outcome: Optional[str] = None,
        probability: Optional[float] = None,
        reasoning: Optional[str] = None,
        format_ok: bool = False,
        raw: str = "",
    ) -> None:
        self.ok = ok
        self.reason = reason
        self.outcome = outcome
        self.probability = probability
        self.reasoning = reasoning
        self.format_ok = format_ok
        self.raw = raw

    def as_dict(self) -> Dict[str, object]:
        return {
            "ok": self.ok,
            "format_ok": self.format_ok,
            "outcome": self.outcome,
            "probability": self.probability,
            "reasoning": self.reasoning,
            "reason": self.reason,
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            "ParseResult(ok={0!r}, format_ok={1!r}, outcome={2!r}, probability={3!r}, "
            "reason={4!r})".format(
                self.ok, self.format_ok, self.outcome, self.probability, self.reason
            )
        )


def _fail(reason: str, raw: str, reasoning: Optional[str] = None) -> ParseResult:
    return ParseResult(ok=False, format_ok=False, reason=reason, reasoning=reasoning, raw=raw)


# --------------------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------------------
_THINK_RE = re.compile(r"<\s*think\s*>(.*?)<\s*/\s*think\s*>", re.DOTALL | re.IGNORECASE)
_THINK_OPEN_RE = re.compile(r"<\s*think\s*>", re.IGNORECASE)
_THINK_CLOSE_RE = re.compile(r"<\s*/\s*think\s*>", re.IGNORECASE)

_ANSWER_RE = re.compile(r"<\s*answer\s*>(.*?)<\s*/\s*answer\s*>", re.DOTALL | re.IGNORECASE)
_ANSWER_OPEN_RE = re.compile(r"<\s*answer\s*>", re.IGNORECASE)
_ANSWER_CLOSE_RE = re.compile(r"<\s*/\s*answer\s*>", re.IGNORECASE)

_VERDICT_KEYWORD_RE = re.compile(r"verdict\s*:", re.IGNORECASE)
_EXPLANATION_KEYWORD_RE = re.compile(r"explanation\s*:", re.IGNORECASE)
_A_GT_B_RE = re.compile(r"\[\s*\[\s*A\s*>\s*B\s*\]\s*\]", re.IGNORECASE)
_B_GT_A_RE = re.compile(r"\[\s*\[\s*B\s*>\s*A\s*\]\s*\]", re.IGNORECASE)

_PROB_RE = re.compile(r"<\s*prob_rb_better\s*>(.*?)<\s*/\s*prob_rb_better\s*>", re.DOTALL | re.IGNORECASE)
_PROB_OPEN_RE = re.compile(r"<\s*prob_rb_better\s*>", re.IGNORECASE)
_PROB_CLOSE_RE = re.compile(r"<\s*/\s*prob_rb_better\s*>", re.IGNORECASE)

#: Characters stripped from the inside of an ``<answer>`` payload before the letter check.
_LETTER_STRIP_CHARS = " \t\r\n[](){}<>\"'`*.,:;!"


def _strip_letter(payload: str) -> str:
    """Return the bare letter of an answer payload, or ``""`` when it is not one letter.

    ``"[[A]]"``, ``" A "`` and ``"[b]"`` all reduce to a single letter; ``"A>B"`` and
    ``"A is better"`` reduce to something longer and are therefore rejected.
    """
    text = payload.strip()
    for _ in range(4):  # tolerate stacked wrappers such as "[[A]]"
        stripped = text.strip(_LETTER_STRIP_CHARS)
        if stripped == text:
            break
        text = stripped
    if len(text) == 1 and text.upper() in ("A", "B"):
        return text.upper()
    return ""


def _extract_think(text: str, want_block: bool = True):
    """Return ``(reasoning, error_or_None)`` for the ``<think>`` block."""
    blocks = _THINK_RE.findall(text)
    if want_block:
        if not blocks:
            if _THINK_OPEN_RE.search(text):
                return None, UNCLOSED_THINK
            if _THINK_CLOSE_RE.search(text):
                return None, UNCLOSED_THINK
            return None, MISSING_THINK
        if len(blocks) > 1:
            return None, MULTIPLE_THINK_BLOCKS
        body = blocks[0].strip()
        if not body:
            return None, EMPTY_THINK
        return body, None
    # Block optional: still reject a broken one rather than ignoring it.
    opens = len(_THINK_OPEN_RE.findall(text))
    closes = len(_THINK_CLOSE_RE.findall(text))
    if opens != closes:
        return None, UNCLOSED_THINK
    if len(blocks) > 1:
        return None, MULTIPLE_THINK_BLOCKS
    return (blocks[0].strip() or None) if blocks else None, None


def _extract_letter_answer(text: str, reasoning: Optional[str]):
    """Return ``(letter, error_or_None)`` for an unambiguous ``<answer>`` tag."""
    match = _ANSWER_RE.search(text)
    matches = list(_ANSWER_RE.finditer(text))
    if len(matches) > 1:
        letters = set()
        for m in matches:
            letters.add(_strip_letter(m.group(1)))
        # Two tags carrying the same letter are still malformed: the model was asked for
        # exactly one verdict and a duplicated tag usually signals a rewritten answer.
        if len(letters) != 1 or "" in letters:
            return None, MULTIPLE_ANSWER_TAGS
        return None, MULTIPLE_ANSWER_TAGS
    if match is None:
        opens = len(_ANSWER_OPEN_RE.findall(text))
        closes = len(_ANSWER_CLOSE_RE.findall(text))
        if opens or closes:
            return None, UNCLOSED_ANSWER_TAG
        return None, NO_ANSWER_TAG
    letter = _strip_letter(match.group(1))
    if not letter:
        return None, INVALID_ANSWER_LETTER
    return letter, None


# --------------------------------------------------------------------------------------
# vanilla -- ``<answer>A</answer>``
# --------------------------------------------------------------------------------------
def parse_vanilla(text: str) -> ParseResult:
    if not isinstance(text, str):
        raise TypeError("completion must be a str")
    if not text.strip():
        return _fail(EMPTY_OUTPUT, text)
    letter, err = _extract_letter_answer(text, None)
    if err is not None:
        return _fail(err, text)
    return ParseResult(ok=True, format_ok=True, outcome=letter, reason=OK, raw=text)


# --------------------------------------------------------------------------------------
# j1 -- ``<think>...</think><answer>[[A]]</answer>``
# --------------------------------------------------------------------------------------
def parse_j1(text: str) -> ParseResult:
    if not isinstance(text, str):
        raise TypeError("completion must be a str")
    if not text.strip():
        return _fail(EMPTY_OUTPUT, text)

    reasoning, think_err = _extract_think(text, want_block=True)
    if think_err is not None:
        return _fail(think_err, text)

    letter, err = _extract_letter_answer(text, reasoning)
    if err is not None:
        return _fail(err, text, reasoning=reasoning)

    # The paper's seed prompt asks for the bracketed letter; a bare letter still proves
    # the verdict is correct, so ``ok`` stays true and only ``format_ok`` degrades.
    answer_span = _ANSWER_RE.search(text)
    bracketed = bool(re.match(r"\s*\[\s*\[", answer_span.group(1))) if answer_span else False

    if text.rfind("</think>") > text.find("<answer>"):
        return ParseResult(
            ok=True,
            format_ok=False,
            outcome=letter,
            reasoning=reasoning,
            reason=THINK_AFTER_ANSWER,
            raw=text,
        )

    return ParseResult(
        ok=True,
        format_ok=bracketed,
        outcome=letter,
        reasoning=reasoning,
        reason=OK if bracketed else ANSWER_NOT_BRACKETED,
        raw=text,
    )


# --------------------------------------------------------------------------------------
# j4r -- ``Explanation: ...`` / ``Verdict: [[A>B]]``
# --------------------------------------------------------------------------------------
def parse_j4r(text: str) -> ParseResult:
    if not isinstance(text, str):
        raise TypeError("completion must be a str")
    if not text.strip():
        return _fail(EMPTY_OUTPUT, text)

    verdict_hits = list(_VERDICT_KEYWORD_RE.finditer(text))
    if not verdict_hits:
        return _fail(NO_VERDICT_KEYWORD, text)
    if len(verdict_hits) > 1:
        return _fail(MULTIPLE_VERDICT_KEYWORDS, text)

    tail = text[verdict_hits[0].end():]
    a_hits = _A_GT_B_RE.findall(tail)
    b_hits = _B_GT_A_RE.findall(tail)
    if a_hits and b_hits:
        return _fail(CONFLICTING_VERDICTS, text)
    if not a_hits and not b_hits:
        return _fail(NO_VERDICT_TAG, text)
    if len(a_hits) + len(b_hits) > 1:
        return _fail(CONFLICTING_VERDICTS, text)

    outcome = "A" if a_hits else "B"
    explanation = text[: verdict_hits[0].start()]
    has_explanation = bool(explanation.strip()) and bool(_EXPLANATION_KEYWORD_RE.search(explanation))
    return ParseResult(
        ok=True,
        format_ok=has_explanation,
        outcome=outcome,
        reason=OK if has_explanation else MISSING_EXPLANATION,
        raw=text,
    )


# --------------------------------------------------------------------------------------
# jwc -- ``<think>...</think><prob_rb_better>0.85</prob_rb_better>``
# --------------------------------------------------------------------------------------
def parse_jwc(text: str) -> ParseResult:
    if not isinstance(text, str):
        raise TypeError("completion must be a str")
    if not text.strip():
        return _fail(EMPTY_OUTPUT, text)

    reasoning, think_err = _extract_think(text, want_block=True)
    if think_err is not None:
        return _fail(think_err, text)

    matches = list(_PROB_RE.finditer(text))
    if not matches:
        if _PROB_OPEN_RE.search(text) or _PROB_CLOSE_RE.search(text):
            return _fail(UNCLOSED_PROB_TAG, text, reasoning=reasoning)
        return _fail(NO_PROB_SECTION, text, reasoning=reasoning)
    if len(matches) > 1:
        return _fail(MULTIPLE_PROB_TAGS, text, reasoning=reasoning)

    payload = matches[0].group(1).strip()
    try:
        value = float(payload)
    except (TypeError, ValueError):
        return _fail(PROB_NOT_A_NUMBER, text, reasoning=reasoning)
    if not math.isfinite(value):
        return _fail(PROB_NOT_FINITE, text, reasoning=reasoning)
    if value < 0.0 or value > 1.0:
        return _fail(PROB_OUT_OF_RANGE, text, reasoning=reasoning)

    return ParseResult(
        ok=True,
        format_ok=True,
        probability=value,
        reasoning=reasoning,
        reason=OK,
        raw=text,
    )


# --------------------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------------------
PARSERS = {
    "vanilla": parse_vanilla,
    "j1": parse_j1,
    "j4r": parse_j4r,
    "jwc": parse_jwc,
}


def get_parser(output_format: str):
    try:
        return PARSERS[output_format]
    except KeyError:
        raise ValueError(
            "unknown output_format {0!r}; expected one of {1}".format(
                output_format, tuple(sorted(PARSERS))
            )
        )


def parse(output_format: str, text: str) -> ParseResult:
    """Dispatch on an output-format name (typically ``config.output_format``)."""
    return get_parser(output_format)(text)


__all__ = [
    "PARSERS",
    "ParseResult",
    "REASON_CODES",
    "get_parser",
    "parse",
    "parse_j1",
    "parse_j4r",
    "parse_jwc",
    "parse_vanilla",
] + list(REASON_CODES)
