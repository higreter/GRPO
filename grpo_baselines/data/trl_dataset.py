"""Turn method-neutral :class:`data.pair_views.PairView` records into TRL rows.

TRL's ``GRPOTrainer`` has three requirements that this module satisfies in one
place, so no trainer has to remember them:

1. ``prompt`` must be a **conversational** value -- ``is_conversational`` in
   ``trl/data_utils.py`` only detects ``[{"role": ..., "content": ...}, ...]``,
   anything else falls back to the plain-text tokeniser path and the chat
   template would silently be skipped.  :func:`build_trl_row` therefore stores
   the output of :func:`common.prompts.build_messages`, never a pre-rendered
   string.
2. Every extra column is forwarded verbatim to the reward function
   (``_calculate_rewards`` copies all keys except
   ``prompt``/``completion``/``completion_ids``).  So the row carries
   ``ground_truth``, ``pair_id``, ``sample_id``, ``is_swapped`` and ``view`` --
   the reward functions need all of them.
3. Rows must not carry ``images``/``image`` keys, otherwise TRL treats them as a
   multimodal dataset.

The rendering itself lives in :mod:`common.prompts`; this module only assembles
rows and can be imported without any third-party package.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Iterator, List, Optional

from common import prompts
from common.labels import ANSWER_A, ANSWER_B


class DatasetRowError(ValueError):
    """Raised when a pair-view record cannot be turned into a consistent TRL row."""


#: Columns forwarded to the reward function.  ``prompt`` is handled separately
#: and ``answer_a``/``answer_b`` are kept for the per-sample evaluation dump.
REWARD_COLUMNS = ("sample_id", "pair_id", "ground_truth", "is_swapped", "view",
                  "group_id", "source_line", "source_dataset")

#: Keys that must never appear in a TRL row (they switch on multimodal paths).
FORBIDDEN_COLUMNS = ("images", "image", "pixel_values", "image_grid_thw")


def build_trl_row(record: Dict[str, Any], method: str) -> Dict[str, Any]:
    """Convert one pair-view record into a TRL dataset row.

    ``record`` is the output of :meth:`data.pair_views.PairView.to_record`, so it
    already carries both display answers and the display-space ground truth.
    The conversation is rebuilt from ``question``/``answer_a``/``answer_b`` --
    never from the source dataset's ``text`` column, which holds the old
    JudgeLRM rationale rather than this project's prompt.
    """
    prompts.check_method(method)
    for key in ("sample_id", "pair_id", "question", "answer_a", "answer_b",
                "ground_truth"):
        if key not in record:
            raise DatasetRowError("record is missing required key {0!r}".format(key))
    ground_truth = record["ground_truth"]
    if ground_truth not in (ANSWER_A, ANSWER_B):
        raise DatasetRowError(
            "ground_truth must be {0!r} or {1!r}, got {2!r}".format(
                ANSWER_A, ANSWER_B, ground_truth)
        )
    if record["answer_a"] == record["answer_b"]:
        raise DatasetRowError(
            "sample {0!r} has identical answers in both slots; a judge cannot learn "
            "from a degenerate pair".format(record["sample_id"])
        )

    row: Dict[str, Any] = {
        "prompt": prompts.build_messages(
            method,
            question=record["question"],
            response_a=record["answer_a"],
            response_b=record["answer_b"],
        ),
        "ground_truth": ground_truth,
        "sample_id": record["sample_id"],
        "pair_id": record["pair_id"],
        "is_swapped": bool(record.get("is_swapped", False)),
        "view": record.get("view") or ("swapped" if record.get("is_swapped") else "original"),
        "answer_a": record["answer_a"],
        "answer_b": record["answer_b"],
        # Traceability (brief Sec. 3.3).  Both are optional because an eval view built
        # from RewardBench has no HelpSteer group; `None` is the honest value there.
        "group_id": record.get("group_id"),
        "source_line": record.get("source_line"),
        "source_dataset": record.get("source_dataset"),
    }
    for key in FORBIDDEN_COLUMNS:
        if key in row:  # pragma: no cover - defensive
            raise DatasetRowError("row must not carry multimodal column {0!r}".format(key))
    return row


def iter_trl_rows(records: Iterable[Dict[str, Any]], method: str) -> Iterator[Dict[str, Any]]:
    """Stream TRL rows, checking that ids stay unique."""
    seen = set()
    for record in records:
        row = build_trl_row(record, method)
        if row["sample_id"] in seen:
            raise DatasetRowError("duplicate sample_id {0!r}".format(row["sample_id"]))
        seen.add(row["sample_id"])
        yield row


def build_trl_rows(records: Iterable[Dict[str, Any]], method: str,
                   limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Materialise TRL rows in memory (used by the trainers and by the tests)."""
    rows: List[Dict[str, Any]] = []
    for row in iter_trl_rows(records, method):
        rows.append(row)
        if limit is not None and len(rows) >= limit:
            break
    return rows


def assert_row_invariants(rows: List[Dict[str, Any]], *, states_per_sample: int) -> Dict[str, Any]:
    """Check the properties the paired methods depend on.

    Returns a small summary so a run manifest can record what was verified.  Any
    violation is a hard error: a silently broken pairing would turn J1/J4R into
    plain GRPO while still being reported as the paired method.
    """
    if not rows:
        raise DatasetRowError("no rows to check")
    by_pair: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for row in rows:
        by_pair.setdefault(row["pair_id"], {})[row["view"]] = row
    views = sorted({row["view"] for row in rows})
    expected_views = ["original", "swapped"] if states_per_sample == 2 else ["original"]
    if views != expected_views:
        raise DatasetRowError(
            "expected views {0} for states_per_sample={1}, found {2}".format(
                expected_views, states_per_sample, views)
        )
    if states_per_sample == 2:
        incomplete = [pair_id for pair_id, members in by_pair.items()
                      if set(members) != {"original", "swapped"}]
        if incomplete:
            raise DatasetRowError(
                "{0} pair(s) are missing a state, e.g. {1!r}".format(
                    len(incomplete), incomplete[0])
            )
        mismatched = []
        for pair_id, members in by_pair.items():
            original, swapped = members["original"], members["swapped"]
            if original["answer_a"] != swapped["answer_b"] or original["answer_b"] != swapped["answer_a"]:
                mismatched.append(pair_id)
            elif original["ground_truth"] == swapped["ground_truth"]:
                mismatched.append(pair_id)
        if mismatched:
            raise DatasetRowError(
                "{0} pair(s) do not swap answers and flip the ground truth, e.g. "
                "{1!r}".format(len(mismatched), mismatched[0])
            )
    return {
        "rows": len(rows),
        "pairs": len(by_pair),
        "views": views,
        "states_per_sample": states_per_sample,
        "ground_truth_counts": {
            ANSWER_A: sum(1 for row in rows if row["ground_truth"] == ANSWER_A),
            ANSWER_B: sum(1 for row in rows if row["ground_truth"] == ANSWER_B),
        },
    }


def build_hf_dataset(rows: List[Dict[str, Any]], purpose: str = "GRPO training"):
    """Wrap rows in a ``datasets.Dataset`` (imported lazily).

    Kept separate from :func:`build_trl_rows` so every CPU test can exercise the
    row construction without the ``datasets`` package, which is not installed in
    the read-only environment.
    """
    from trainers.trl_bridge import require

    datasets = require("datasets", purpose)
    return datasets.Dataset.from_list(rows)


__all__ = [
    "DatasetRowError",
    "FORBIDDEN_COLUMNS",
    "REWARD_COLUMNS",
    "assert_row_invariants",
    "build_hf_dataset",
    "build_trl_row",
    "build_trl_rows",
    "iter_trl_rows",
]
