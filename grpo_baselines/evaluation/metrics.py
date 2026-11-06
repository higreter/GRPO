"""Metric definitions for the RewardBench evaluation of the four baselines.

Every metric here is computed from *per-sample records*, never from pre-aggregated
numbers, so the same function serves the live evaluator and a post-hoc re-analysis of a
saved JSONL.  The record schema is documented in :mod:`evaluation.rewardbench`.

Three distinctions drive the design, because collapsing them is exactly how judge
evaluations end up overstating themselves:

1. **Parser success vs. accuracy.**  ``parser_success_rate`` and
   ``invalid_output_rate`` are reported next to accuracy; nothing is silently dropped.
   Accuracy is reported twice: ``accuracy_on_parsed`` (denominator = parsable rows) and
   ``accuracy_all_treated_wrong`` (denominator = every row, unparsable counts as wrong).
   A method cannot buy accuracy by refusing to answer.
2. **Both-order accuracy vs. position consistency.**  *Both-order accuracy* asks whether
   the judge got both orders right -- a joint success event.  *Position consistency* asks
   whether the judge picked the same physical response in both orders, regardless of
   whether that response was the correct one.  A judge that always answers "A" is
   perfectly position-consistent and 50%-ish accurate, so the two numbers must be
   published separately.
3. **Calibration.**  Brier/ECE/MCE are defined only for the probability head of JwC.
   An A/B-only judge has no probability to calibrate; assigning it a synthetic 1.0
   confidence would make its Brier score depend on the label balance rather than on any
   forecast, so :func:`calibration_metrics` refuses to invent one (see
   ``require_probabilities``).
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "ECE_BINS",
    "InvalidRecordError",
    "accuracy_report",
    "OFFICIAL_CATEGORIES",
    "aggregate_subsets",
    "both_order_accuracy",
    "brier_score",
    "calibration_metrics",
    "expected_calibration_error",
    "full_report",
    "maximum_calibration_error",
    "official_aggregate",
    "pair_views",
    "position_consistency",
    "subset_accuracy",
]

#: Equal-width bins used by JwC App. C.1.  Kept as a module constant so the report and the
#: code cannot disagree about how many bins were used.
ECE_BINS = 10

#: Tolerance for "these two completion texts are the same answer" comparisons is not
#: needed anywhere: identity is decided by the display-order mapping, never by text.


class InvalidRecordError(ValueError):
    """A per-sample record is missing a field the metric needs."""


# --------------------------------------------------------------------------------------
# record helpers
# --------------------------------------------------------------------------------------
def _require(record: Dict[str, Any], key: str) -> Any:
    if key not in record:
        raise InvalidRecordError("record is missing required field {0!r}".format(key))
    return record[key]


def pair_views(records: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Group per-sample records into ``{qid: {"original": rec, "swapped": rec}}``.

    Raises if a qid appears twice for the same view: a duplicated view would double-count
    a sample in every aggregate, and the evaluator is required to keep *all* test rows,
    so silent de-duplication is not an option.
    """
    grouped: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for record in records:
        qid = str(_require(record, "qid"))
        is_swapped = bool(_require(record, "is_swapped"))
        bucket = grouped.setdefault(qid, {})
        view = "swapped" if is_swapped else "original"
        if view in bucket:
            raise InvalidRecordError(
                "qid {0!r} appears twice for the {1} view".format(qid, view))
        bucket[view] = record
    return grouped


def _parsed_correct(record: Dict[str, Any]) -> Optional[bool]:
    """Correctness of one row, or ``None`` when the output did not parse."""
    parsed = record.get("parsed_ok")
    if parsed is None:
        # Fall back to the stored outcome so a hand-written record still works.
        parsed = record.get("outcome") is not None
    if not parsed:
        return None
    correct = record.get("correct")
    if correct is None:
        raise InvalidRecordError(
            "parsed record for qid {0!r} has no 'correct' flag".format(record.get("qid")))
    return bool(correct)


# --------------------------------------------------------------------------------------
# single-view metrics
# --------------------------------------------------------------------------------------
def accuracy_report(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Accuracy, parser success and invalid rate for one ordering.

    ``records`` must all be the same view (all original, or all swapped); mixing them
    would compare a ground truth against the wrong display order.
    """
    if not records:
        raise InvalidRecordError("cannot compute accuracy over zero records")
    views = {bool(_require(record, "is_swapped")) for record in records}
    if len(views) > 1:
        raise InvalidRecordError(
            "accuracy_report received mixed original/swapped rows; split them first")

    total = len(records)
    parsed = 0
    correct = 0
    for record in records:
        verdict = _parsed_correct(record)
        if verdict is None:
            continue
        parsed += 1
        correct += int(verdict)

    return {
        "view": "swapped" if views.pop() else "original",
        "count": total,
        "parsed": parsed,
        "invalid": total - parsed,
        "parser_success_rate": parsed / total,
        "invalid_output_rate": (total - parsed) / total,
        "correct": correct,
        "accuracy_on_parsed": (correct / parsed) if parsed else None,
        "accuracy_all_treated_wrong": correct / total,
    }


def position_consistency(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Fraction of pairs where both orders chose the *same physical response*.

    Physical identity is what matters: under a swap the correct letter flips, so
    "same letter" is the wrong test and a judge that answers "A" twice is maximally
    inconsistent with itself only if the two orders are mapped back first.

    Unparsable rows are counted in the denominator as inconsistent, and the count of
    pairs that could not be judged is reported separately -- the rate is never computed
    over a self-selected subset.
    """
    grouped = pair_views(records)
    total = 0
    consistent = 0
    both_parsed = 0
    for qid, views in sorted(grouped.items()):
        if "original" not in views or "swapped" not in views:
            continue
        original, swapped = views["original"], views["swapped"]
        total += 1
        original_ok = _parsed_correct(original) is not None
        swapped_ok = _parsed_correct(swapped) is not None
        if not (original_ok and swapped_ok):
            continue
        both_parsed += 1
        chosen = original.get("chosen_response")
        other = swapped.get("chosen_response")
        if chosen is None or other is None:
            raise InvalidRecordError(
                "qid {0!r} lacks chosen_response; position consistency needs the mapped "
                "physical answer, not the display letter".format(qid))
        if str(chosen) == str(other):
            consistent += 1
    if not total:
        raise InvalidRecordError("no complete pairs to score for position consistency")
    return {
        "pairs": total,
        "pairs_both_parsed": both_parsed,
        "consistent": consistent,
        "position_consistency": consistent / total,
        "position_consistency_on_both_parsed": (consistent / both_parsed) if both_parsed else None,
    }


def both_order_accuracy(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Fraction of pairs where *both* orderings were judged correctly.

    This is the J1 consistency event (arXiv:2505.10320 Sec. 2.2) and also the strictest
    reading of J4R's two-state requirement.  Counting it separately from
    :func:`position_consistency` keeps "consistently wrong" from looking like skill.
    """
    grouped = pair_views(records)
    total = 0
    both_correct = 0
    original_only = 0
    swapped_only = 0
    neither = 0
    for qid, views in sorted(grouped.items()):
        if "original" not in views or "swapped" not in views:
            continue
        total += 1
        first = _parsed_correct(views["original"]) is True
        second = _parsed_correct(views["swapped"]) is True
        if first and second:
            both_correct += 1
        elif first:
            original_only += 1
        elif second:
            swapped_only += 1
        else:
            neither += 1
    if not total:
        raise InvalidRecordError("no complete pairs to score for both-order accuracy")
    return {
        "pairs": total,
        "both_correct": both_correct,
        "original_only": original_only,
        "swapped_only": swapped_only,
        "neither": neither,
        "both_order_accuracy": both_correct / total,
    }


# --------------------------------------------------------------------------------------
# per-subset aggregation
# --------------------------------------------------------------------------------------
def subset_accuracy(records: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Accuracy per official subset, computed on the original ordering.

    RewardBench's leaderboard reports the original order, so that is the number quoted
    per subset; the swapped ordering is carried in the same report under a separate key
    so both are visible.
    """
    by_subset: Dict[str, List[Dict[str, Any]]] = {}
    for record in records:
        subset = str(_require(record, "source_subset"))
        by_subset.setdefault(subset, []).append(record)

    report: Dict[str, Dict[str, Any]] = {}
    for subset, rows in sorted(by_subset.items()):
        original = [row for row in rows if not _require(row, "is_swapped")]
        swapped = [row for row in rows if _require(row, "is_swapped")]
        entry: Dict[str, Any] = {"pairs": len(original) or len(swapped)}
        if original:
            entry.update({
                "original_" + key: value
                for key, value in accuracy_report(original).items()
                if key in ("count", "accuracy_on_parsed", "accuracy_all_treated_wrong",
                           "parser_success_rate")
            })
        if swapped:
            entry.update({
                "swapped_" + key: value
                for key, value in accuracy_report(swapped).items()
                if key in ("count", "accuracy_on_parsed", "accuracy_all_treated_wrong",
                           "parser_success_rate")
            })
            entry.update(both_order_accuracy(original + swapped))
            entry.update(position_consistency(original + swapped))
        report[subset] = entry
    return report


#: RewardBench's published aggregation: the mean of the four top-level categories, each
#: itself the mean of its subsets.  Hard-coded from the official split so the aggregate
#: cannot drift when a subset happens to be missing from a run.
OFFICIAL_CATEGORIES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("Chat", ("alpacaeval-easy", "alpacaeval-hard", "alpacaeval-length", "mt-bench-easy",
              "mt-bench-med", "mt-bench-hard")),
    ("Chat Hard", ("llmbar-adver-GPTInst", "llmbar-adver-GPTOut", "llmbar-adver-manual",
                   "llmbar-adver-neighbor", "llmbar-natural")),
    ("Safety", ("refusals-dangerous", "refusals-offensive", "xstest-should-refuse",
                "xstest-should-respond", "donotanswer")),
    ("Reasoning", ("hep-cpp", "hep-go", "hep-java", "hep-js", "hep-python", "hep-rust",
                   "math-prm")),
)


def official_aggregate(per_subset: Dict[str, Dict[str, Any]], *,
                       key: str = "original_accuracy_on_parsed") -> Dict[str, Any]:
    """RewardBench's four-category aggregate plus the overall mean.

    ``key`` selects which accuracy to aggregate, so the report can show the official
    number (original order, parsable rows) next to a stricter variant without
    recomputing the split.  A missing subset is reported explicitly instead of being
    skipped: silently averaging over whatever survived is how a broken run looks healthy.
    """
    categories: Dict[str, float] = {}
    missing: List[str] = []
    for name, subsets in OFFICIAL_CATEGORIES:
        values = []
        for subset in subsets:
            entry = per_subset.get(subset)
            if entry is None or entry.get(key) is None:
                missing.append(subset)
                continue
            values.append(float(entry[key]))
        if values:
            categories[name] = sum(values) / len(values)
    if not categories:
        raise InvalidRecordError(
            "no subset provided {0!r}; cannot compute the official aggregate".format(key))
    overall = sum(categories.values()) / len(categories)
    return {
        "key": key,
        "categories": categories,
        "average": overall,
        "missing_subsets": missing,
        "complete": not missing,
    }


def _carries(per_subset: Dict[str, Dict[str, Any]], key: str) -> bool:
    """True when at least one subset actually has this metric."""
    return any(isinstance(entry, dict) and entry.get(key) is not None
               for entry in per_subset.values())


def aggregate_subsets(per_subset: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """All three official aggregates in one call, plus pair-level summaries.

    A run over a single ordering has no ``swapped_*`` numbers at all, and a single-view
    report must not be an error: the missing aggregate is reported as unavailable instead
    of raising, because "we did not run that ordering" and "we had no data for it" are
    different statements and the caller needs to be able to tell them apart.
    """
    report: Dict[str, Any] = {"subsets": len(per_subset)}
    for name, key in (
        ("official_original", "original_accuracy_on_parsed"),
        ("official_swapped", "swapped_accuracy_on_parsed"),
        ("official_both_order", "both_order_accuracy"),
    ):
        if _carries(per_subset, key):
            report[name] = official_aggregate(per_subset, key=key)
        else:
            report[name] = {
                "available": False,
                "key": key,
                "reason": "this run contains no rows carrying {0!r}; a missing ordering "
                          "cannot be averaged".format(key),
            }
    return report


# --------------------------------------------------------------------------------------
# calibration (JwC only)
# --------------------------------------------------------------------------------------
def brier_score(probabilities: Sequence[float], targets: Sequence[int]) -> float:
    """``(1/N) * sum (p_i - y_i)^2`` -- JwC App. C.1, the *loss*, not the reward.

    Note the sign convention: the reward is ``1 - brier`` (JwC Eq. 1), the metric is the
    brier itself, so a lower number is better here.
    """
    if len(probabilities) != len(targets):
        raise InvalidRecordError("probabilities and targets differ in length")
    if not probabilities:
        raise InvalidRecordError("cannot compute a Brier score over zero samples")
    total = 0.0
    for probability, target in zip(probabilities, targets):
        value = float(probability)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise InvalidRecordError("probability {0!r} is not in [0, 1]".format(probability))
        if int(target) not in (0, 1):
            raise InvalidRecordError("target must be 0 or 1, got {0!r}".format(target))
        total += (value - int(target)) ** 2
    return total / len(probabilities)


def _binned(probabilities: Sequence[float], targets: Sequence[int]
            ) -> List[Tuple[int, int, float, float]]:
    """Return ``(bin_index, count, mean_confidence, accuracy)`` for non-empty bins."""
    bins: List[List[float]] = [[] for _ in range(ECE_BINS)]
    hits: List[List[int]] = [[] for _ in range(ECE_BINS)]
    for probability, target in zip(probabilities, targets):
        value = float(probability)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise InvalidRecordError("probability {0!r} is not in [0, 1]".format(probability))
        # p == 1.0 belongs to the last bin, not an out-of-range 10th index.
        index = min(int(value * ECE_BINS), ECE_BINS - 1)
        bins[index].append(value)
        hits[index].append(int(target))
    rows = []
    for index in range(ECE_BINS):
        if not bins[index]:
            continue
        count = len(bins[index])
        confidence = sum(bins[index]) / count
        accuracy = sum(hits[index]) / count
        rows.append((index, count, confidence, accuracy))
    return rows


def expected_calibration_error(probabilities: Sequence[float], targets: Sequence[int], *,
                               bins: int = ECE_BINS) -> float:
    """``sum_k (|B_k| / N) * |acc(B_k) - conf(B_k)|`` (JwC App. C.1, K = 10 equal width)."""
    if bins != ECE_BINS:
        raise InvalidRecordError(
            "ECE is specified with {0} equal-width bins in JwC App. C.1; {1} would not be "
            "comparable to the paper".format(ECE_BINS, bins))
    if len(probabilities) != len(targets):
        raise InvalidRecordError("probabilities and targets differ in length")
    if not probabilities:
        raise InvalidRecordError("cannot compute ECE over zero samples")
    total = len(probabilities)
    error = 0.0
    for _index, count, confidence, accuracy in _binned(probabilities, targets):
        error += (count / total) * abs(accuracy - confidence)
    return error


def maximum_calibration_error(probabilities: Sequence[float], targets: Sequence[int]) -> float:
    """``max_k |acc(B_k) - conf(B_k)|`` over non-empty bins (worst-case miscalibration)."""
    if len(probabilities) != len(targets):
        raise InvalidRecordError("probabilities and targets differ in length")
    if not probabilities:
        raise InvalidRecordError("cannot compute MCE over zero samples")
    worst = 0.0
    for _index, _count, confidence, accuracy in _binned(probabilities, targets):
        worst = max(worst, abs(accuracy - confidence))
    return worst


def calibration_metrics(records: Sequence[Dict[str, Any]], *,
                        method: str) -> Dict[str, Any]:
    """Brier / ECE / MCE for a probability-emitting method.

    ``method`` must be ``"jwc"``.  For an A/B-only method this raises
    :class:`InvalidRecordError` rather than fabricating confidence: forcing a hard judge
    to 100% confidence and then comparing calibration against a model that actually emits
    probabilities is not a fair comparison, it is a different metric.  The report states
    this explicitly.

    Rows with a ``Tie`` ground truth are skipped, matching JwC App. C.1 (a tie is not a
    binary target).  Rows whose probability failed to parse are excluded from the
    calibration numbers themselves but counted in ``excluded_unparsable`` so the reader
    knows the denominator.
    """
    if method != "jwc":
        raise InvalidRecordError(
            "calibration metrics are defined for the probability head only; method {0!r} "
            "emits a hard A/B verdict and has no confidence to calibrate".format(method))

    probabilities: List[float] = []
    targets: List[int] = []
    skipped_tie = 0
    excluded_unparsable = 0
    for record in records:
        target = record.get("probability_target")
        if target is None:
            if str(record.get("ground_truth", "")).upper() == "TIE":
                skipped_tie += 1
                continue
            raise InvalidRecordError(
                "record for qid {0!r} has no probability_target".format(record.get("qid")))
        if record.get("probability") is None:
            excluded_unparsable += 1
            continue
        probabilities.append(float(record["probability"]))
        targets.append(int(target))

    if not probabilities:
        raise InvalidRecordError(
            "no parsable probabilities; the calibration metrics would be undefined")

    return {
        "method": method,
        "count": len(probabilities),
        "brier": brier_score(probabilities, targets),
        "ece": expected_calibration_error(probabilities, targets),
        "mce": maximum_calibration_error(probabilities, targets),
        "bins": ECE_BINS,
        "skipped_tie": skipped_tie,
        "excluded_unparsable": excluded_unparsable,
        "mean_probability": sum(probabilities) / len(probabilities),
        "positive_rate": sum(targets) / len(targets),
    }


def full_report(records: Sequence[Dict[str, Any]], *, method: str) -> Dict[str, Any]:
    """Everything the report quotes, computed from one list of per-sample records."""
    originals = [row for row in records if not _require(row, "is_swapped")]
    swapped = [row for row in records if _require(row, "is_swapped")]
    per_subset = subset_accuracy(records)
    report: Dict[str, Any] = {
        "method": method,
        "samples": len(records),
        "qids": len(pair_views(records)),
        "original": accuracy_report(originals) if originals else None,
        "swapped": accuracy_report(swapped) if swapped else None,
        "per_subset": per_subset,
        "aggregate": aggregate_subsets(per_subset),
    }
    if originals and swapped:
        joined = list(originals) + list(swapped)
        report["both_order"] = both_order_accuracy(joined)
        report["position_consistency"] = position_consistency(joined)
    if method == "jwc":
        report["calibration"] = calibration_metrics(records, method=method)
    else:
        report["calibration"] = {
            "available": False,
            "reason": "hard A/B verdict; no probability to calibrate "
                      "(see evaluation.metrics.calibration_metrics)",
        }
    return report
