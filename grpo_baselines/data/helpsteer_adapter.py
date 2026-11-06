"""Adapter: ``database/helpsteer3_train_clean.jsonl`` -> training views.

Read-only source.  Nothing here rewrites, filters or subsamples the file; the only
transformation is schema mapping plus the (method-dependent) construction of the
swapped state, which is a *prompt* transformation and duplicates the pair rather than
dropping anything.

Verified field semantics (see the preparation report and database/README.md)
---------------------------------------------------------------------------
``label == "A"``  <=>  ``answer1_body`` is the better response   (11068 rows)
``label == "B"``  <=>  ``answer2_body`` is the better response   (11122 rows)
all 22190 rows have ``view == "original"``.

The dataset's ``text`` column holds legacy JudgeLRM ``<think>`` explanations.  It is
**not** a prompt template and is never used here; the templates live in
:mod:`common.prompts` and come from each paper.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, Iterator, List, Optional

from common import labels, paths
from common.reproducibility import iter_jsonl

from .pair_views import PairView, build_views

SOURCE_DATASET_NAME = "helpsteer3_train_clean"

#: Columns the adapter requires; anything else in the JSONL is ignored.
REQUIRED_SOURCE_FIELDS = (
    "question_body",
    "answer1_body",
    "answer2_body",
    "label",
    "group_id",
    "view",
    "source_line",
)

#: Columns that must never reach a training row (legacy explanations, raw labels).
FORBIDDEN_OUTPUT_FIELDS = ("text", "source_overall_preference")


class AdapterError(ValueError):
    """Raised when a source row violates the documented contract."""


def _require_str(record: Dict[str, Any], key: str, where: str) -> str:
    value = record.get(key)
    if not isinstance(value, str):
        raise AdapterError("{0}: field {1!r} must be a str, got {2!r}".format(
            where, key, type(value).__name__))
    if not value.strip():
        raise AdapterError("{0}: field {1!r} is empty".format(where, key))
    return value


def convert_row(record: Dict[str, Any], row_index: int) -> Dict[str, Any]:
    """Validate one source row and return the intermediate pair description."""
    where = "helpsteer row {0}".format(row_index)
    missing = [key for key in REQUIRED_SOURCE_FIELDS if key not in record]
    if missing:
        raise AdapterError("{0}: missing fields {1}".format(where, missing))

    question = _require_str(record, "question_body", where)
    answer_1 = _require_str(record, "answer1_body", where)
    answer_2 = _require_str(record, "answer2_body", where)
    source_label = _require_str(record, "label", where).strip()

    if record.get("view") not in (None, "original"):
        raise AdapterError(
            "{0}: unexpected view {1!r}; the clean file is documented to contain only "
            "'original' rows".format(where, record.get("view"))
        )

    try:
        better_response = labels.source_label_to_response(source_label)
    except ValueError as exc:
        raise AdapterError("{0}: {1}".format(where, exc))

    source_line = record.get("source_line")
    group_id = record.get("group_id")
    pair_id = "hs3-{0}".format(
        source_line if source_line not in (None, "") else "row{0}".format(row_index)
    )
    metadata = {
        "source_label": source_label,
        "source_line": source_line,
        "group_id": group_id,
        "domain": record.get("domain"),
        "language": record.get("language"),
        "token_len": record.get("token_len"),
    }
    return {
        "pair_id": pair_id,
        "source_row_index": row_index,
        "question": question,
        "answer_1": answer_1,
        "answer_2": answer_2,
        "better_response": better_response,
        "source_dataset": SOURCE_DATASET_NAME,
        "metadata": metadata,
    }


def build_pair_views(pair: Dict[str, Any], *, two_states: bool) -> List[PairView]:
    """Expand one converted pair into its prompt state(s)."""
    return build_views(
        pair_id=pair["pair_id"],
        source_row_index=pair["source_row_index"],
        question=pair["question"],
        answer_1=pair["answer_1"],
        answer_2=pair["answer_2"],
        better_response=pair["better_response"],
        source_dataset=pair["source_dataset"],
        two_states=two_states,
        metadata=pair.get("metadata"),
    )


def iter_pairs(
    path: Optional[str] = None,
    *,
    limit: Optional[int] = None,
    source_row_indices: Optional[Iterable[int]] = None,
) -> Iterator[Dict[str, Any]]:
    """Yield converted pairs from the clean training file (streaming, no full load).

    ``source_row_indices`` is solely for an auditable smoke-test selection of real rows
    (for example the measured longest row).  Formal runs leave it as ``None`` and stream
    every source pair without rewriting or filtering the source dataset.
    """
    source = path or str(paths.HELPSTEER_TRAIN)
    if not os.path.exists(source):
        raise FileNotFoundError("training file not found: {0}".format(source))
    selected = None
    if source_row_indices is not None:
        selected = set(source_row_indices)
        if not selected or any(isinstance(index, bool) or not isinstance(index, int) or index < 0
                               for index in selected):
            raise AdapterError("source_row_indices must contain one or more non-negative integers")
    emitted = 0
    for index, record in enumerate(iter_jsonl(source)):
        if selected is not None and index not in selected:
            continue
        if limit is not None and emitted >= limit:
            return
        yield convert_row(record, index)
        emitted += 1
        if selected is not None and emitted == len(selected):
            return
    if selected is not None and emitted != len(selected):
        raise AdapterError(
            "requested {0} source row(s), but found only {1} in {2}".format(
                len(selected), emitted, source)
        )


def iter_training_rows(
    path: Optional[str] = None,
    *,
    two_states: bool,
    limit: Optional[int] = None,
    source_row_indices: Optional[Iterable[int]] = None,
) -> Iterator[Dict[str, Any]]:
    """Yield training rows (one per prompt state) in the minimal project schema."""
    for pair in iter_pairs(path, limit=limit, source_row_indices=source_row_indices):
        for view in build_pair_views(pair, two_states=two_states):
            record = view.to_record()
            for forbidden in FORBIDDEN_OUTPUT_FIELDS:
                if forbidden in record:  # pragma: no cover - defensive
                    raise AssertionError(
                        "training row leaked forbidden field {0!r}".format(forbidden)
                    )
            yield record


def materialise(
    output_path: str,
    *,
    two_states: bool,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Write the training rows to ``output_path`` and return a summary.

    Only the summary is derived here; the caller decides where to put it.
    """
    paths.assert_writable_file(output_path)
    rows = 0
    pairs = set()
    per_view: Dict[str, int] = {}
    per_ground_truth: Dict[str, int] = {}
    with open(output_path, "w", encoding="utf-8") as handle:
        for record in iter_training_rows(two_states=two_states, limit=limit):
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
            rows += 1
            pairs.add(record["pair_id"])
            per_view[record["view"]] = per_view.get(record["view"], 0) + 1
            key = record["ground_truth"]
            per_ground_truth[key] = per_ground_truth.get(key, 0) + 1
    summary = {
        "output_path": output_path,
        "two_states": two_states,
        "rows": rows,
        "pairs": len(pairs),
        "rows_per_view": per_view,
        "rows_per_ground_truth": per_ground_truth,
        "source": str(paths.HELPSTEER_TRAIN),
        "source_dataset": SOURCE_DATASET_NAME,
    }
    if two_states:
        if rows != 2 * len(pairs):
            raise AssertionError("two-state expansion lost rows: {0} rows, {1} pairs".format(
                rows, len(pairs)))
        if per_ground_truth.get(labels.ANSWER_A, 0) != per_ground_truth.get(labels.ANSWER_B, 0):
            raise AssertionError(
                "swapped states must balance the label distribution: {0}".format(
                    per_ground_truth)
            )
    elif rows != len(pairs):
        raise AssertionError("single-state expansion produced {0} rows for {1} pairs".format(
            rows, len(pairs)))
    return summary


def scan_source(path: Optional[str] = None) -> Dict[str, Any]:
    """Read-only census of the training file (counts, no transformation)."""
    source = path or str(paths.HELPSTEER_TRAIN)
    total = 0
    by_label: Dict[str, int] = {}
    by_view: Dict[str, int] = {}
    by_language: Dict[int, int] = {}
    for index, record in enumerate(iter_jsonl(source)):
        total += 1
        convert_row(record, index)  # validation only
        label = record["label"]
        by_label[label] = by_label.get(label, 0) + 1
        view = record.get("view")
        by_view[view] = by_view.get(view, 0) + 1
        language = record.get("language")
        by_language[language] = by_language.get(language, 0) + 1
    return {
        "path": source,
        "rows": total,
        "by_label": by_label,
        "by_view": by_view,
        "by_language": by_language,
    }


__all__ = [
    "AdapterError",
    "FORBIDDEN_OUTPUT_FIELDS",
    "REQUIRED_SOURCE_FIELDS",
    "SOURCE_DATASET_NAME",
    "build_pair_views",
    "convert_row",
    "iter_pairs",
    "iter_training_rows",
    "materialise",
    "scan_source",
]
