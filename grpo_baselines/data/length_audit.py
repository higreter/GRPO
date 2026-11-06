"""Length audit with the *real* Qwen tokenizer and each method's *real* prompt template.

The audit answers three questions the implementation report must answer with numbers:

1. what is the prompt-length distribution (min/p50/p90/p95/p99/max) for each method;
2. how many samples exceed ``max_prompt_length`` and by how much;
3. does ``max_prompt_length + max_completion_length`` still fit inside the model's
   32,768-token context window.

Nothing is truncated or dropped silently: a sample that does not fit is *reported* in an
auditable list, and :func:`assert_within_limits` raises when the configured limits would
force the trainer to drop or truncate it.  Only ``transformers`` is required, and only for
:func:`load_tokenizer`; the statistics helpers are pure standard library so they can be
unit-tested without the model.

Chat-template note: prompts are rendered through
``tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)``,
i.e. exactly the string the trainer will tokenise.  Measuring hand-concatenated text would
miss the ChatML control tokens and understate every length.
"""

from __future__ import annotations

import json
import os
import statistics
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from common import paths, prompts

PERCENTILES = (50.0, 90.0, 95.0, 99.0)


class LengthAuditError(ValueError):
    """Raised when configured limits would require truncation or dropping."""


# --------------------------------------------------------------------------------------
# Pure statistics (no model needed)
# --------------------------------------------------------------------------------------
def percentile(sorted_values: Sequence[int], q: float) -> float:
    """Linear-interpolation percentile (``numpy.percentile`` default, "linear")."""
    if not sorted_values:
        raise ValueError("cannot take a percentile of an empty sequence")
    if not 0.0 <= q <= 100.0:
        raise ValueError("percentile must be in [0, 100], got {0!r}".format(q))
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = (len(sorted_values) - 1) * (q / 100.0)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return float(sorted_values[lower]) + fraction * (
        float(sorted_values[upper]) - float(sorted_values[lower])
    )


def summarise(lengths: Sequence[int]) -> Dict[str, Any]:
    """min/p50/p90/p95/p99/max/mean/std plus the raw count."""
    if not lengths:
        raise ValueError("no lengths to summarise")
    ordered = sorted(int(value) for value in lengths)
    summary: Dict[str, Any] = {
        "count": len(ordered),
        "min": ordered[0],
        "max": ordered[-1],
        "mean": round(statistics.fmean(ordered), 2),
        "std": round(statistics.pstdev(ordered), 2) if len(ordered) > 1 else 0.0,
    }
    for q in PERCENTILES:
        key = "p{0}".format(int(q))
        summary[key] = int(round(percentile(ordered, q)))
    return summary


def count_exceeding(lengths: Sequence[int], limit: int) -> Dict[str, Any]:
    """Count (and quantify) samples longer than ``limit``."""
    if limit <= 0:
        raise ValueError("limit must be positive")
    exceeding = [int(value) for value in lengths if int(value) > limit]
    total = len(lengths)
    worst = max(exceeding) if exceeding else None
    return {
        "limit": limit,
        "total": total,
        "exceeding": len(exceeding),
        "exceeding_ratio": round(len(exceeding) / total, 6) if total else 0.0,
        "max_exceeding": worst,
        "max_overshoot": (worst - limit) if worst is not None else 0,
    }


def audit_lengths(lengths: Sequence[int], max_prompt_length: int,
                  max_completion_length: int,
                  *, context_limit: int = paths.MAX_CONTEXT_TOKENS) -> Dict[str, Any]:
    """Full audit record for one method/config pair."""
    if max_prompt_length + max_completion_length > context_limit:
        raise LengthAuditError(
            "max_prompt_length + max_completion_length = {0} > context limit {1}".format(
                max_prompt_length + max_completion_length, context_limit)
        )
    summary = summarise(lengths)
    overflow = count_exceeding(lengths, max_prompt_length)
    within = [value for value in lengths if value <= max_prompt_length]
    kept_summary = summarise(within) if within else None
    return {
        "prompt_tokens": summary,
        "max_prompt_length": max_prompt_length,
        "max_completion_length": max_completion_length,
        "completion_headroom_after_p95": max_prompt_length - summary["p95"],
        "context_limit": context_limit,
        "context_budget": max_prompt_length + max_completion_length,
        "overflow_at_max_prompt_length": overflow,
        "within_limit": kept_summary,
        "would_need_truncation": overflow["exceeding"] > 0,
    }


def assert_within_limits(audit: Dict[str, Any]) -> None:
    """Raise when the audit says samples would be truncated or dropped."""
    overflow = audit["overflow_at_max_prompt_length"]
    if overflow["exceeding"]:
        raise LengthAuditError(
            "{0} of {1} prompts ({2:.2%}) exceed max_prompt_length={3}; the longest is "
            "{4} tokens (+{5}). Refusing to truncate or drop silently -- raise "
            "max_prompt_length or document an explicit, audited policy.".format(
                overflow["exceeding"], overflow["total"], overflow["exceeding_ratio"],
                overflow["limit"], overflow["max_exceeding"], overflow["max_overshoot"])
        )


# --------------------------------------------------------------------------------------
# Tokenizer-backed measurement
# --------------------------------------------------------------------------------------
def load_tokenizer(model_path: Optional[str] = None) -> Any:
    """Load the fixed project tokenizer (no network access, no model weights)."""
    from transformers import AutoTokenizer  # noqa: WPS433 - lazy, keeps module importable

    path = model_path or str(paths.BASE_MODEL)
    return AutoTokenizer.from_pretrained(path, use_fast=True, local_files_only=True)


def render_prompt(tokenizer: Any, method: str, record: Dict[str, Any]) -> str:
    """Render one dataset record into the exact prompt string the trainer will see."""
    messages = prompts.build_messages(
        method,
        question=record["question"],
        response_a=record["answer_a"],
        response_b=record["answer_b"],
    )
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def measure(
    tokenizer: Any,
    method: str,
    records: Iterable[Dict[str, Any]],
    *,
    max_rows: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Measure the prompt length of every record; return per-record measurements."""
    results: List[Dict[str, Any]] = []
    for index, record in enumerate(records):
        if max_rows is not None and index >= max_rows:
            break
        text = render_prompt(tokenizer, method, record)
        results.append(
            {
                "sample_id": record.get("sample_id") or record.get("qid"),
                "pair_id": record.get("pair_id"),
                "view": record.get("view") or ("swapped" if record.get("is_swapped") else "original"),
                "prompt_tokens": len(tokenizer(text, add_special_tokens=False)["input_ids"]),
                "characters": len(text),
            }
        )
    if not results:
        raise LengthAuditError("no records were measured for method {0!r}".format(method))
    return results


def run_audit(
    tokenizer: Any,
    method: str,
    records: Iterable[Dict[str, Any]],
    max_prompt_length: int,
    max_completion_length: int,
    *,
    max_rows: Optional[int] = None,
    oversize_sample_limit: int = 25,
) -> Dict[str, Any]:
    """Measure + summarise + list the oversize samples, in one auditable record."""
    measurements = measure(tokenizer, method, records, max_rows=max_rows)
    lengths = [item["prompt_tokens"] for item in measurements]
    audit = audit_lengths(lengths, max_prompt_length, max_completion_length)
    oversize = sorted(
        (item for item in measurements if item["prompt_tokens"] > max_prompt_length),
        key=lambda item: item["prompt_tokens"],
        reverse=True,
    )
    audit.update(
        {
            "method": method,
            "rows_measured": len(measurements),
            "oversize_samples": oversize[:oversize_sample_limit],
            "oversize_samples_truncated": len(oversize) > oversize_sample_limit,
            "oversize_total": len(oversize),
        }
    )
    return audit


def write_audit(path: str, payload: Dict[str, Any]) -> None:
    # A file, not a directory: `assert_writable_output` would mkdir the path and the
    # failure would surface later as IsADirectoryError from open().
    paths.assert_writable_file(path)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


__all__ = [
    "LengthAuditError",
    "assert_within_limits",
    "audit_lengths",
    "count_exceeding",
    "load_tokenizer",
    "measure",
    "percentile",
    "render_prompt",
    "run_audit",
    "summarise",
    "write_audit",
]
