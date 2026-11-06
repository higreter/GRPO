"""Adapter: ``database/rewardbench_v1_ood_clean.jsonl`` -> out-of-domain evaluation views.

Read-only source, evaluation only.  **No RewardBench row may become a training row**;
:func:`assert_not_training` exists so a caller that mixes the two up fails loudly.

Verified structure of the clean file (2985 rows, 23 subsets)
-----------------------------------------------------------
* every row carries exactly one ``view``: ``chosen_A`` (1495 rows) or ``chosen_B`` (1490);
* ``gt_class == label`` on every row and equals the display letter holding the chosen
  response (verified: 0 mismatches);
* ``answer1_body`` is display **A**, ``answer2_body`` is display **B** for that row,
  and the chosen response occupies the slot named by ``gt_class`` (verified: 0
  mismatches against ``source_chosen``).

Because the stored file already fixes one display order per qid, the *other* order is
constructed here by exchanging the two answers and flipping the ground-truth letter.  Both
orders are returned for every qid, which is what the dual-view evaluation protocol needs.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Iterable, Iterator, List, Optional

from common import labels, paths
from common.reproducibility import iter_jsonl

SOURCE_DATASET_NAME = "rewardbench_v1"

REQUIRED_SOURCE_FIELDS = (
    "qid",
    "source_subset",
    "question_body",
    "answer1_body",
    "answer2_body",
    "gt_class",
    "view",
)

STORED_VIEW_CHOSEN_A = "chosen_A"
STORED_VIEW_CHOSEN_B = "chosen_B"
STORED_VIEWS = (STORED_VIEW_CHOSEN_A, STORED_VIEW_CHOSEN_B)

#: The 23 source_subset values of RewardBench v1, grouped the way the official leaderboard
#: groups them.  Written out because the evaluator validates every row against this list:
#: a subset that is spelled differently in the clean file would otherwise vanish from the
#: per-subset table and quietly shrink the official aggregate.
#:
#: Verified against rewardbench_v1_ood_clean.jsonl (2,985 rows) -- the comment after each
#: entry is the measured row count, so a future re-clean that changes the split shows up as
#: a mismatch here rather than as a silently different denominator.
OFFICIAL_SUBSETS = (
    # Chat
    "alpacaeval-easy",          # 100
    "alpacaeval-hard",          # 95
    "alpacaeval-length",        # 95
    "mt-bench-easy",            # 28
    "mt-bench-med",             # 40
    "mt-bench-hard",            # 37
    # Chat Hard
    "llmbar-adver-GPTInst",     # 92
    "llmbar-adver-GPTOut",      # 47
    "llmbar-adver-manual",      # 46
    "llmbar-adver-neighbor",    # 134
    "llmbar-natural",           # 100
    # Safety
    "refusals-dangerous",       # 100
    "refusals-offensive",       # 100
    "xstest-should-refuse",     # 154
    "xstest-should-respond",    # 250
    "donotanswer",              # 136
    # Reasoning
    "hep-cpp",                  # 164
    "hep-go",                   # 164
    "hep-java",                 # 164
    "hep-js",                   # 164
    "hep-python",               # 164
    "hep-rust",                 # 164
    "math-prm",                 # 447
)


class RewardBenchAdapterError(ValueError):
    """Raised when a RewardBench row violates the documented contract."""


def _require_str(record: Dict[str, Any], key: str, where: str) -> str:
    value = record.get(key)
    if not isinstance(value, str):
        raise RewardBenchAdapterError("{0}: field {1!r} must be a str, got {2!r}".format(
            where, key, type(value).__name__))
    if not value.strip():
        raise RewardBenchAdapterError("{0}: field {1!r} is empty".format(where, key))
    return value


def convert_row(record: Dict[str, Any], row_index: int) -> Dict[str, Any]:
    """Validate one stored row and return it in the project's display-space schema."""
    where = "rewardbench row {0}".format(row_index)
    missing = [key for key in REQUIRED_SOURCE_FIELDS if key not in record]
    if missing:
        raise RewardBenchAdapterError("{0}: missing fields {1}".format(where, missing))

    qid = _require_str(record, "qid", where)
    subset = _require_str(record, "source_subset", where)
    question = _require_str(record, "question_body", where)
    answer_1 = _require_str(record, "answer1_body", where)
    answer_2 = _require_str(record, "answer2_body", where)
    gt_class = _require_str(record, "gt_class", where)
    stored_view = _require_str(record, "view", where)

    if gt_class not in (labels.ANSWER_A, labels.ANSWER_B):
        raise RewardBenchAdapterError("{0}: gt_class must be A or B, got {1!r}".format(
            where, gt_class))
    if stored_view not in STORED_VIEWS:
        raise RewardBenchAdapterError("{0}: unexpected view {1!r}".format(where, stored_view))
    if subset not in OFFICIAL_SUBSETS:
        raise RewardBenchAdapterError(
            "{0}: subset {1!r} is not one of the 23 official RewardBench v1 subsets; a "
            "mis-spelled subset would drop out of the per-subset table and skew the "
            "official aggregate".format(where, subset)
        )
    expected_view = STORED_VIEW_CHOSEN_A if gt_class == labels.ANSWER_A else STORED_VIEW_CHOSEN_B
    if stored_view != expected_view:
        raise RewardBenchAdapterError(
            "{0}: gt_class={1!r} contradicts stored view {2!r}".format(
                where, gt_class, stored_view)
        )

    return {
        "qid": qid,
        "source_row_index": row_index,
        "source_subset": subset,
        "question": question,
        "answer_display_a": answer_1,
        "answer_display_b": answer_2,
        "ground_truth_stored": gt_class,
        "stored_view": stored_view,
        "chosen_model": record.get("chosen_model"),
        "rejected_model": record.get("rejected_model"),
        "source_dataset": SOURCE_DATASET_NAME,
    }


def build_eval_views(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return ``[original, swapped]`` evaluation views for one converted row.

    ``original`` is the order stored in the clean file; ``swapped`` exchanges the two
    answers and flips the ground truth.  ``answer_a``/``answer_b`` are always display
    order, ``ground_truth`` is always display space, and ``is_swapped`` records which
    view is which so the evaluator can map both back onto the same physical response.
    """
    original = {
        "qid": record["qid"],
        "view_id": "{0}::original".format(record["qid"]),
        "source_subset": record["source_subset"],
        "question": record["question"],
        "answer_a": record["answer_display_a"],
        "answer_b": record["answer_display_b"],
        "ground_truth": record["ground_truth_stored"],
        "is_swapped": False,
        "source_row_index": record["source_row_index"],
        "source_dataset": record["source_dataset"],
        "chosen_model": record.get("chosen_model"),
        "rejected_model": record.get("rejected_model"),
    }
    swapped = dict(original)
    swapped["view_id"] = "{0}::swapped".format(record["qid"])
    swapped["answer_a"] = record["answer_display_b"]
    swapped["answer_b"] = record["answer_display_a"]
    swapped["ground_truth"] = labels.flip_answer(record["ground_truth_stored"])
    swapped["is_swapped"] = True
    # Invariant: both views hold the same physical content, only the order differs.
    if {original["answer_a"], original["answer_b"]} != {swapped["answer_a"], swapped["answer_b"]}:
        raise AssertionError("swapped evaluation view changed the answer content")
    return [original, swapped]


def iter_records(path: Optional[str] = None, *,
                 limit: Optional[int] = None) -> Iterator[Dict[str, Any]]:
    source = path or str(paths.REWARDBENCH_OOD)
    if not os.path.exists(source):
        raise FileNotFoundError("RewardBench file not found: {0}".format(source))
    for index, record in enumerate(iter_jsonl(source)):
        if limit is not None and index >= limit:
            return
        yield convert_row(record, index)


def iter_eval_views(path: Optional[str] = None, *,
                    limit: Optional[int] = None) -> Iterator[Dict[str, Any]]:
    """Yield both display orders for every qid (2 views per stored row)."""
    for record in iter_records(path, limit=limit):
        for view in build_eval_views(record):
            yield view


def load_all_views(path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Materialise every evaluation view in memory (2985 rows -> 5970 views)."""
    views = list(iter_eval_views(path))
    qids = {view["qid"] for view in views}
    if len(views) != 2 * len(qids):
        raise AssertionError("expected exactly two views per qid, got {0} views for {1} qids"
                             .format(len(views), len(qids)))
    return views


def scan_source(path: Optional[str] = None) -> Dict[str, Any]:
    """Read-only census of the evaluation file."""
    source = path or str(paths.REWARDBENCH_OOD)
    total = 0
    by_subset: Dict[str, int] = {}
    by_gt: Dict[str, int] = {}
    by_view: Dict[str, int] = {}
    qids = set()
    for index, record in enumerate(iter_jsonl(source)):
        converted = convert_row(record, index)
        total += 1
        qids.add(converted["qid"])
        by_subset[converted["source_subset"]] = by_subset.get(converted["source_subset"], 0) + 1
        gt = converted["ground_truth_stored"]
        by_gt[gt] = by_gt.get(gt, 0) + 1
        view = converted["stored_view"]
        by_view[view] = by_view.get(view, 0) + 1
    unknown = sorted(set(by_subset) - set(OFFICIAL_SUBSETS))
    return {
        "path": source,
        "rows": total,
        "unique_qids": len(qids),
        "subsets": len(by_subset),
        "by_subset": by_subset,
        "by_ground_truth": by_gt,
        "by_stored_view": by_view,
        "unknown_subsets": unknown,
    }


def assert_not_training(path: str) -> None:
    """Refuse to treat the evaluation file as a training source."""
    resolved = os.path.realpath(path)
    if resolved == os.path.realpath(str(paths.REWARDBENCH_OOD)):
        raise ValueError(
            "RewardBench is the out-of-domain *evaluation* set and must never be used for "
            "training, hyper-parameter selection, template choice or checkpoint selection"
        )


__all__ = [
    "OFFICIAL_SUBSETS",
    "REQUIRED_SOURCE_FIELDS",
    "RewardBenchAdapterError",
    "SOURCE_DATASET_NAME",
    "STORED_VIEWS",
    "assert_not_training",
    "build_eval_views",
    "convert_row",
    "iter_eval_views",
    "iter_records",
    "load_all_views",
    "scan_source",
]
