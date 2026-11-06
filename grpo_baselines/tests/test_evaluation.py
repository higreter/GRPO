"""CPU tests for the RewardBench evaluation layer (prompt Sec. 12, Evaluation row).

Three properties are load-bearing here and each gets its own test class:

* every prediction row keeps the full frozen schema, including the raw completion and the
  parse reason, so a failed parse can be audited instead of silently dropped;
* ``both_order_accuracy`` and ``position_consistency`` are computed *separately* -- a judge
  that always picks the worse physical answer is perfectly position-consistent and 0%
  both-order accurate, and collapsing the two would make that look like skill;
* calibration (Brier/ECE/MCE) is only defined for the probability head; asking for it on a
  hard A/B verdict must raise rather than invent 100% confidence.

Everything runs on the standard library.  No model is loaded, no GPU is touched: the
end-to-end test passes a stub ``generate`` callable, which is exactly the seam the runner
was designed around.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest

from common import labels, parsing, paths

from evaluation import metrics, rewardbench


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def _view(qid, *, is_swapped, ground_truth, subset="alpacaeval-easy"):
    """A minimal evaluation view in the shape ``rewardbench_adapter`` produces."""
    order = labels.display_order(is_swapped)
    bodies = {labels.RESPONSE_1: "response one", labels.RESPONSE_2: "response two"}
    return {
        "qid": str(qid),
        "view_id": "{0}::{1}".format(qid, "swapped" if is_swapped else "original"),
        "source_subset": subset,
        "question": "Which response is better?",
        "answer_a": bodies[order[0]],
        "answer_b": bodies[order[1]],
        "ground_truth": ground_truth,
        "is_swapped": is_swapped,
        "source_row_index": 0,
        "source_dataset": "rewardbench_v1",
        "chosen_model": "model-a",
        "rejected_model": "model-b",
    }


def _record(qid, *, is_swapped, ground_truth, outcome, chosen_response,
            parsed_ok=True, correct=None, subset="alpacaeval-easy",
            probability=None, probability_target=None, reason=None, completion="x",
            method="vanilla"):
    """A hand-built prediction row, so metric tests never depend on a parser's mood."""
    if correct is None and parsed_ok and outcome is not None:
        correct = labels.is_correct(outcome, ground_truth)
    return {
        "qid": str(qid),
        "view_id": "{0}::{1}".format(qid, "swapped" if is_swapped else "original"),
        "source_subset": subset,
        "is_swapped": bool(is_swapped),
        "display_order": list(labels.display_order(is_swapped)),
        "ground_truth": ground_truth,
        "question": "q",
        "answer_a": "a",
        "answer_b": "b",
        "raw_completion": completion,
        "parsed_ok": bool(parsed_ok),
        "parse_reason": reason or (parsing.OK if parsed_ok else parsing.NO_ANSWER_TAG),
        "format_ok": bool(parsed_ok),
        "outcome": outcome,
        "chosen_response": chosen_response if parsed_ok else None,
        "probability": probability,
        "probability_target": probability_target,
        "probability_decision": None,
        "correct": correct if parsed_ok else None,
        "latency_seconds": 0.0,
        "method": method,
        "model_path": "",
    }


def _pair(qid, *, better="answer1", original_letter=None, swapped_letter=None,
          subset="alpacaeval-easy", parsed=True):
    """Build the two prediction rows of one qid from display letters."""
    original_gt = labels.ground_truth_for_view(better, False)
    swapped_gt = labels.ground_truth_for_view(better, True)
    rows = []
    for is_swapped, letter, gt in ((False, original_letter, original_gt),
                                   (True, swapped_letter, swapped_gt)):
        ok = parsed and letter is not None
        rows.append(_record(
            qid,
            is_swapped=is_swapped,
            ground_truth=gt,
            outcome=letter if ok else None,
            chosen_response=(labels.display_to_response(letter, is_swapped) if ok else None),
            parsed_ok=ok,
            subset=subset,
        ))
    return rows


class _TempDir(object):
    """A scratch directory that lives *inside* the project root (the only writable root)."""

    def __enter__(self):
        self.path = tempfile.mkdtemp(prefix="_test_eval_", dir=str(paths.RUNS_DIR))
        return self.path

    def __exit__(self, *exc):
        shutil.rmtree(self.path, ignore_errors=True)
        return False


# --------------------------------------------------------------------------------------
# 1. per-sample schema
# --------------------------------------------------------------------------------------
class PredictionSchemaTest(unittest.TestCase):
    def test_every_method_produces_all_frozen_fields(self):
        completions = {
            "vanilla": "<answer>A</answer>",
            "j1": "<think>compare</think><answer>[[A]]</answer>",
            "j4r": "Explanation: compare\nVerdict: [[A>B]]",
            "jwc": "<think>compare</think><prob_rb_better>0.80</prob_rb_better>",
        }
        for method, completion in completions.items():
            for is_swapped in (False, True):
                view = _view("q{0}".format(method), is_swapped=is_swapped,
                             ground_truth=labels.ANSWER_A)
                record = rewardbench.build_prediction_record(
                    method=method, view=view, completion=completion,
                    latency_seconds=0.25, model_path="/tmp/ckpt")
                for field in rewardbench.PREDICTION_FIELDS:
                    self.assertIn(field, record, "{0} missing {1}".format(method, field))
                self.assertEqual(set(record), set(rewardbench.PREDICTION_FIELDS))
                self.assertEqual(record["method"], method)
                self.assertEqual(record["model_path"], "/tmp/ckpt")
                self.assertAlmostEqual(record["latency_seconds"], 0.25)
                self.assertEqual(record["raw_completion"], completion)

    def test_chosen_response_maps_back_to_the_physical_answer(self):
        view_original = _view("q1", is_swapped=False, ground_truth=labels.ANSWER_A)
        view_swapped = _view("q1", is_swapped=True, ground_truth=labels.ANSWER_A)
        first = rewardbench.build_prediction_record(
            method="vanilla", view=view_original,
            completion="<answer>A</answer>", latency_seconds=0.0)
        second = rewardbench.build_prediction_record(
            method="vanilla", view=view_swapped,
            completion="<answer>A</answer>", latency_seconds=0.0)
        # Same display letter, different physical answer -- the swap must be undone.
        self.assertEqual(first["outcome"], "A")
        self.assertEqual(second["outcome"], "A")
        self.assertEqual(first["chosen_response"], labels.RESPONSE_1)
        self.assertEqual(second["chosen_response"], labels.RESPONSE_2)
        self.assertNotEqual(first["chosen_response"], second["chosen_response"])

    def test_display_order_records_the_slot_layout(self):
        original = rewardbench.build_prediction_record(
            method="vanilla", view=_view("q2", is_swapped=False,
                                         ground_truth=labels.ANSWER_A),
            completion="<answer>A</answer>", latency_seconds=0.0)
        swapped = rewardbench.build_prediction_record(
            method="vanilla", view=_view("q2", is_swapped=True,
                                         ground_truth=labels.ANSWER_A),
            completion="<answer>A</answer>", latency_seconds=0.0)
        self.assertEqual(original["display_order"],
                         [labels.RESPONSE_1, labels.RESPONSE_2])
        self.assertEqual(swapped["display_order"],
                         [labels.RESPONSE_2, labels.RESPONSE_1])

    def test_probability_target_only_exists_for_the_probability_head(self):
        view = _view("q3", is_swapped=False, ground_truth=labels.ANSWER_B)
        vanilla = rewardbench.build_prediction_record(
            method="vanilla", view=view, completion="<answer>B</answer>",
            latency_seconds=0.0)
        jwc = rewardbench.build_prediction_record(
            method="jwc", view=view,
            completion="<think>t</think><prob_rb_better>0.9</prob_rb_better>",
            latency_seconds=0.0)
        self.assertIsNone(vanilla["probability"])
        self.assertIsNone(vanilla["probability_target"])
        self.assertIsNone(vanilla["probability_decision"])
        self.assertAlmostEqual(jwc["probability"], 0.9)
        self.assertEqual(jwc["probability_target"], 1)

    def test_an_unparsable_completion_keeps_its_row_and_its_reason(self):
        record = rewardbench.build_prediction_record(
            method="vanilla",
            view=_view("q4", is_swapped=False, ground_truth=labels.ANSWER_A),
            completion="I refuse to answer.",
            latency_seconds=0.0)
        self.assertFalse(record["parsed_ok"])
        self.assertEqual(record["parse_reason"], parsing.NO_ANSWER_TAG)
        self.assertIsNone(record["outcome"])
        self.assertIsNone(record["chosen_response"])
        self.assertIsNone(record["correct"])
        # The raw text is kept, so a human can see what the model actually said.
        self.assertEqual(record["raw_completion"], "I refuse to answer.")


# --------------------------------------------------------------------------------------
# 2. single-view accuracy
# --------------------------------------------------------------------------------------
class AccuracyReportTest(unittest.TestCase):
    def test_accuracy_is_reported_both_with_and_without_invalid_rows(self):
        rows = [
            _record("a", is_swapped=False, ground_truth="A", outcome="A",
                    chosen_response=labels.RESPONSE_1),
            _record("b", is_swapped=False, ground_truth="B", outcome="A",
                    chosen_response=labels.RESPONSE_1),
            _record("c", is_swapped=False, ground_truth="A", outcome=None,
                    chosen_response=None, parsed_ok=False),
            _record("d", is_swapped=False, ground_truth="A", outcome=None,
                    chosen_response=None, parsed_ok=False),
        ]
        report = metrics.accuracy_report(rows)
        self.assertEqual(report["view"], "original")
        self.assertEqual(report["count"], 4)
        self.assertEqual(report["parsed"], 2)
        self.assertEqual(report["invalid"], 2)
        self.assertAlmostEqual(report["parser_success_rate"], 0.5)
        self.assertAlmostEqual(report["invalid_output_rate"], 0.5)
        # Refusing to answer must not be free: the two accuracies differ by design.
        self.assertAlmostEqual(report["accuracy_on_parsed"], 0.5)
        self.assertAlmostEqual(report["accuracy_all_treated_wrong"], 0.25)

    def test_mixed_views_are_rejected_instead_of_averaged(self):
        rows = [
            _record("a", is_swapped=False, ground_truth="A", outcome="A",
                    chosen_response=labels.RESPONSE_1),
            _record("b", is_swapped=True, ground_truth="A", outcome="A",
                    chosen_response=labels.RESPONSE_2),
        ]
        with self.assertRaises(metrics.InvalidRecordError) as ctx:
            metrics.accuracy_report(rows)
        self.assertIn("mixed original/swapped rows", str(ctx.exception))

    def test_empty_batch_is_rejected(self):
        with self.assertRaises(metrics.InvalidRecordError):
            metrics.accuracy_report([])

    def test_duplicate_view_for_one_qid_is_rejected(self):
        row = _record("a", is_swapped=False, ground_truth="A", outcome="A",
                      chosen_response=labels.RESPONSE_1)
        with self.assertRaises(metrics.InvalidRecordError) as ctx:
            metrics.position_consistency([row, dict(row)])
        self.assertIn("appears twice for the original view", str(ctx.exception))


# --------------------------------------------------------------------------------------
# 3. both-order accuracy vs position consistency
# --------------------------------------------------------------------------------------
class BothOrderVersusConsistencyTest(unittest.TestCase):
    """The two metrics must never collapse into one another."""

    def test_position_neutral_judge_is_consistent_and_both_order_correct(self):
        # Always names the better physical answer, which is a different letter per order.
        rows = _pair("q1", better="answer1", original_letter="A", swapped_letter="B")
        self.assertAlmostEqual(metrics.both_order_accuracy(rows)["both_order_accuracy"], 1.0)
        self.assertAlmostEqual(metrics.position_consistency(rows)["position_consistency"], 1.0)

    def test_always_a_judge_is_neither_consistent_nor_both_order_correct(self):
        rows = _pair("q1", better="answer1", original_letter="A", swapped_letter="A")
        both = metrics.both_order_accuracy(rows)
        self.assertAlmostEqual(both["both_order_accuracy"], 0.0)
        self.assertEqual(both["original_only"], 1)
        self.assertAlmostEqual(metrics.position_consistency(rows)["position_consistency"], 0.0)

    def test_consistently_wrong_judge_scores_consistency_one_and_both_order_zero(self):
        # Picks the *worse* physical answer in both orders: perfectly self-consistent,
        # never correct.  If these two numbers were fused, this failure would hide.
        rows = _pair("q1", better="answer1", original_letter="B", swapped_letter="A")
        both = metrics.both_order_accuracy(rows)
        consistency = metrics.position_consistency(rows)
        self.assertAlmostEqual(both["both_order_accuracy"], 0.0)
        self.assertEqual(both["neither"], 1)
        self.assertAlmostEqual(consistency["position_consistency"], 1.0)
        self.assertNotEqual(both["both_order_accuracy"], consistency["position_consistency"])

    def test_unparsable_rows_stay_in_the_denominator(self):
        rows = _pair("q1", better="answer1", original_letter="A", swapped_letter=None,
                     parsed=True)
        both = metrics.both_order_accuracy(rows)
        consistency = metrics.position_consistency(rows)
        self.assertEqual(both["pairs"], 1)
        self.assertAlmostEqual(both["both_order_accuracy"], 0.0)
        self.assertEqual(consistency["pairs"], 1)
        self.assertEqual(consistency["pairs_both_parsed"], 0)
        self.assertIsNone(consistency["position_consistency_on_both_parsed"])
        # The headline rate still counts the unreadable judgement against the judge.
        self.assertAlmostEqual(consistency["position_consistency"], 0.0)

    def test_missing_chosen_response_is_an_error_not_a_guess(self):
        rows = _pair("q1", better="answer1", original_letter="A", swapped_letter="B")
        rows[0]["chosen_response"] = None
        with self.assertRaises(metrics.InvalidRecordError) as ctx:
            metrics.position_consistency(rows)
        self.assertIn("lacks chosen_response", str(ctx.exception))

    def test_half_pairs_are_skipped_not_scored(self):
        only_original = [_pair("q1", better="answer1", original_letter="A",
                               swapped_letter="B")[0]]
        with self.assertRaises(metrics.InvalidRecordError) as ctx:
            metrics.both_order_accuracy(only_original)
        self.assertIn("no complete pairs", str(ctx.exception))


# --------------------------------------------------------------------------------------
# 4. per-subset and official aggregation
# --------------------------------------------------------------------------------------
class OfficialAggregateTest(unittest.TestCase):
    def test_missing_subsets_are_reported_and_mark_the_aggregate_incomplete(self):
        per_subset = {
            "alpacaeval-easy": {"original_accuracy_on_parsed": 1.0},
            "alpacaeval-hard": {"original_accuracy_on_parsed": 0.5},
        }
        report = metrics.official_aggregate(per_subset)
        self.assertFalse(report["complete"])
        self.assertIn("mt-bench-easy", report["missing_subsets"])
        self.assertIn("math-prm", report["missing_subsets"])
        # Chat is the only category with any data, so the four-category mean is just Chat.
        self.assertEqual(set(report["categories"]), {"Chat"})
        self.assertAlmostEqual(report["categories"]["Chat"], 0.75)
        self.assertAlmostEqual(report["average"], 0.75)

    def test_empty_input_raises_instead_of_returning_zero(self):
        with self.assertRaises(metrics.InvalidRecordError) as ctx:
            metrics.official_aggregate({})
        self.assertIn("cannot compute the official aggregate", str(ctx.exception))

    def test_all_twenty_three_subsets_produce_a_complete_aggregate(self):
        per_subset = {}
        for _, subsets in metrics.OFFICIAL_CATEGORIES:
            for subset in subsets:
                per_subset[subset] = {"original_accuracy_on_parsed": 0.5}
        report = metrics.official_aggregate(per_subset)
        self.assertTrue(report["complete"])
        self.assertEqual(report["missing_subsets"], [])
        self.assertEqual(len(report["categories"]), 4)
        self.assertAlmostEqual(report["average"], 0.5)

    def test_categories_are_averaged_per_subset_then_across_categories(self):
        # Chat has six subsets, Reasoning seven; a flat row mean would weight the larger
        # category more, so this checks the leak the official split avoids.
        per_subset = {}
        for name, subsets in metrics.OFFICIAL_CATEGORIES:
            for subset in subsets:
                per_subset[subset] = {"original_accuracy_on_parsed":
                                      1.0 if name == "Chat" else 0.0}
        report = metrics.official_aggregate(per_subset)
        self.assertAlmostEqual(report["categories"]["Chat"], 1.0)
        self.assertAlmostEqual(report["categories"]["Reasoning"], 0.0)
        self.assertAlmostEqual(report["average"], 0.25)

    def test_per_subset_report_keeps_both_orders_and_pair_metrics(self):
        rows = []
        for index in range(2):
            rows.extend(_pair("q{0}".format(index), better="answer1",
                              original_letter="A", swapped_letter="B",
                              subset="math-prm"))
        report = metrics.subset_accuracy(rows)
        self.assertEqual(list(report), ["math-prm"])
        entry = report["math-prm"]
        self.assertEqual(entry["pairs"], 2)
        self.assertAlmostEqual(entry["original_accuracy_on_parsed"], 1.0)
        self.assertAlmostEqual(entry["swapped_accuracy_on_parsed"], 1.0)
        self.assertAlmostEqual(entry["both_order_accuracy"], 1.0)
        self.assertAlmostEqual(entry["position_consistency"], 1.0)

    def test_aggregate_subsets_reports_all_three_views(self):
        rows = [_pair("q1", better="answer1", original_letter="A",
                      swapped_letter="A", subset="math-prm")[0],
                _pair("q1", better="answer1", original_letter="A",
                      swapped_letter="A", subset="math-prm")[1]]
        report = metrics.aggregate_subsets(metrics.subset_accuracy(rows))
        self.assertEqual(set(report), {"official_original", "official_swapped",
                                       "official_both_order", "subsets"})
        self.assertEqual(report["subsets"], 1)
        self.assertAlmostEqual(report["official_original"]["average"], 1.0)
        self.assertAlmostEqual(report["official_swapped"]["average"], 0.0)
        self.assertAlmostEqual(report["official_both_order"]["average"], 0.0)


# --------------------------------------------------------------------------------------
# 5. calibration (JwC only)
# --------------------------------------------------------------------------------------
class CalibrationTest(unittest.TestCase):
    def test_calibration_is_refused_for_hard_verdict_methods(self):
        for method in ("vanilla", "j1", "j4r"):
            with self.assertRaises(metrics.InvalidRecordError) as ctx:
                metrics.calibration_metrics([], method=method)
            self.assertIn("probability head only", str(ctx.exception))
            self.assertIn("no confidence to calibrate", str(ctx.exception))

    def test_brier_is_the_loss_not_the_reward(self):
        # Brier = (1/N) sum (p - y)^2; the JwC reward is 1 - Brier.
        self.assertAlmostEqual(metrics.brier_score([0.1, 0.9], [0, 1]), 0.01)
        self.assertAlmostEqual(metrics.brier_score([0.9, 0.1], [0, 1]), 0.81)
        self.assertAlmostEqual(metrics.brier_score([0.5], [0]), 0.25)

    def test_brier_rejects_malformed_input(self):
        with self.assertRaises(metrics.InvalidRecordError):
            metrics.brier_score([0.5], [0, 1])
        with self.assertRaises(metrics.InvalidRecordError):
            metrics.brier_score([], [])

    def test_ece_and_mce_match_a_hand_computation(self):
        probabilities = [0.2, 0.2, 0.7, 0.7]
        targets = [0, 1, 1, 0]
        # bin 2: conf .2, acc .5 -> gap .3, weight 2/4 ; bin 7: conf .7, acc .5 -> gap .2
        ece = metrics.expected_calibration_error(probabilities, targets)
        mce = metrics.maximum_calibration_error(probabilities, targets)
        self.assertAlmostEqual(ece, 0.3 * 0.5 + 0.2 * 0.5)
        self.assertAlmostEqual(mce, 0.3)
        self.assertAlmostEqual(metrics.brier_score(probabilities, targets), 0.315)

    def test_a_perfectly_calibrated_head_has_zero_ece(self):
        # Every populated bin has accuracy == mean confidence, so ECE and MCE vanish.
        # (A handful of singletons would *not* be zero -- each lonely bin contributes its
        # own gap, which is why calibration needs a sample, not four rows.)
        probabilities = [0.2] * 10 + [0.9] * 10
        targets = [0] * 8 + [1] * 2 + [0] * 1 + [1] * 9
        self.assertAlmostEqual(metrics.expected_calibration_error(probabilities, targets), 0.0)
        self.assertAlmostEqual(metrics.maximum_calibration_error(probabilities, targets), 0.0)

    def test_singleton_bins_still_contribute_their_gap(self):
        # 0.05/0.15/0.85/0.95 land in four distinct bins, so ECE is the mean |acc-conf|.
        probabilities = [0.05, 0.15, 0.85, 0.95]
        targets = [0, 0, 1, 1]
        self.assertAlmostEqual(
            metrics.expected_calibration_error(probabilities, targets), 0.1)
        self.assertAlmostEqual(
            metrics.maximum_calibration_error(probabilities, targets), 0.15)

    def test_probability_one_lands_in_the_last_bin(self):
        # Naive ``int(1.0 * 10)`` would index bin 10 and crash; the metric must clamp.
        self.assertAlmostEqual(
            metrics.expected_calibration_error([1.0, 1.0], [1, 1]), 0.0)
        self.assertAlmostEqual(
            metrics.maximum_calibration_error([0.0, 0.0], [1, 1]), 1.0)

    def test_ece_refuses_a_bin_count_that_breaks_comparability(self):
        with self.assertRaises(ValueError):
            metrics.expected_calibration_error([0.5], [1], bins=5)

    def test_ties_are_skipped_and_unparsable_rows_are_counted_separately(self):
        rows = [
            _record("a", is_swapped=False, ground_truth="B", outcome="B",
                    chosen_response=labels.RESPONSE_2, method="jwc",
                    probability=0.9, probability_target=1),
            # Unparsable probability, but the binary target still exists (it comes from
            # the ground truth, not from the completion).
            _record("b", is_swapped=False, ground_truth="A", outcome=None,
                    chosen_response=None, parsed_ok=False, method="jwc",
                    probability=None, probability_target=0),
            # A tie is not a binary target at all, so the row is out of scope entirely.
            _record("d", is_swapped=False, ground_truth="Tie", outcome=None,
                    chosen_response=None, parsed_ok=False, method="jwc",
                    probability=None, probability_target=None),
        ]
        report = metrics.calibration_metrics(rows, method="jwc")
        self.assertEqual(report["count"], 1)
        self.assertEqual(report["skipped_tie"], 1)
        self.assertEqual(report["excluded_unparsable"], 1)
        self.assertAlmostEqual(report["brier"], 0.01)
        self.assertAlmostEqual(report["mean_probability"], 0.9)
        self.assertAlmostEqual(report["positive_rate"], 1.0)

    def test_a_fully_unparsable_probability_batch_raises(self):
        rows = [_record("a", is_swapped=False, ground_truth="A", outcome=None,
                        chosen_response=None, parsed_ok=False, method="jwc",
                        probability=None, probability_target=0)]
        with self.assertRaises(metrics.InvalidRecordError) as ctx:
            metrics.calibration_metrics(rows, method="jwc")
        self.assertIn("no parsable probabilities", str(ctx.exception))


# --------------------------------------------------------------------------------------
# 6. whole-report wiring
# --------------------------------------------------------------------------------------
class FullReportTest(unittest.TestCase):
    def test_hard_verdict_methods_report_calibration_as_unavailable(self):
        rows = _pair("q1", better="answer1", original_letter="A", swapped_letter="B")
        report = metrics.full_report(rows, method="vanilla")
        self.assertEqual(report["method"], "vanilla")
        self.assertEqual(report["samples"], 2)
        self.assertEqual(report["qids"], 1)
        self.assertFalse(report["calibration"]["available"])
        self.assertIn("hard A/B verdict", report["calibration"]["reason"])
        self.assertIn("both_order", report)
        self.assertIn("position_consistency", report)

    def test_probability_head_gets_real_calibration_numbers(self):
        rows = [
            _record("a", is_swapped=False, ground_truth="B", outcome="B",
                    chosen_response=labels.RESPONSE_2, method="jwc",
                    probability=0.8, probability_target=1, subset="math-prm"),
            _record("b", is_swapped=False, ground_truth="A", outcome="B",
                    chosen_response=labels.RESPONSE_2, method="jwc",
                    probability=0.9, probability_target=0, subset="math-prm"),
        ]
        report = metrics.full_report(rows, method="jwc")
        self.assertNotIn("available", report["calibration"])
        self.assertEqual(report["calibration"]["method"], "jwc")
        self.assertIn("brier", report["calibration"])
        self.assertIn("ece", report["calibration"])
        self.assertEqual(report["calibration"]["count"], 2)
        self.assertAlmostEqual(report["calibration"]["positive_rate"], 0.5)
        self.assertIsNone(report["swapped"])
        self.assertNotIn("both_order", report)

    def test_report_json_serialisable(self):
        rows = _pair("q1", better="answer1", original_letter="A", swapped_letter="B")
        json.dumps(metrics.full_report(rows, method="vanilla"), sort_keys=True)


# --------------------------------------------------------------------------------------
# 7. end-to-end runner with a stub model
# --------------------------------------------------------------------------------------
class RunnerTest(unittest.TestCase):
    def _run(self, method, stub, **kwargs):
        with _TempDir() as tmp:
            output = os.path.join(tmp, "predictions.jsonl")
            summary = rewardbench.evaluate(
                method=method, generate=stub, output_path=output, **kwargs)
            records = rewardbench.load_predictions(output)
            self.assertTrue(os.path.exists(output + ".summary.json"))
            with open(output + ".summary.json", encoding="utf-8") as handle:
                on_disk = json.load(handle)
        return summary, records, on_disk

    def test_one_row_per_sample_and_ordering_no_filtering(self):
        # The stub returns unparsable text on purpose: every row must still be written.
        summary, records, on_disk = self._run(
            "vanilla", lambda messages: "no verdict here",
            limit=3, progress_every=0)
        self.assertEqual(len(records), 6)
        self.assertEqual({row["qid"] for row in records}, {row["qid"] for row in records})
        self.assertEqual(len({(row["qid"], row["is_swapped"]) for row in records}), 6)
        self.assertTrue(all(not row["parsed_ok"] for row in records))
        self.assertEqual(summary["samples"], 6)
        self.assertEqual(summary["views_run"], ["original", "swapped"])
        self.assertEqual(on_disk["samples"], 6)
        # Accuracy on zero parsable rows is undefined, not zero.
        self.assertIsNone(summary["original"]["accuracy_on_parsed"])
        self.assertAlmostEqual(summary["original"]["invalid_output_rate"], 1.0)

    def test_the_two_orderings_differ_and_are_both_kept(self):
        summary, records, _ = self._run(
            "vanilla", lambda messages: "<answer>A</answer>",
            limit=2, progress_every=0)
        originals = [row for row in records if not row["is_swapped"]]
        swapped = [row for row in records if row["is_swapped"]]
        self.assertEqual(len(originals), 2)
        self.assertEqual(len(swapped), 2)
        # Always answering "A" is right exactly when the stored order puts the better
        # response in slot A, and wrong in every swapped ordering.  That asymmetry is the
        # whole point of the dual-view protocol, so assert it from the rows themselves
        # rather than from an assumed ground truth.
        expected_original = sum(1 for row in originals if row["ground_truth"] == "A") / 2.0
        expected_swapped = sum(1 for row in swapped if row["ground_truth"] == "A") / 2.0
        self.assertAlmostEqual(summary["original"]["accuracy_on_parsed"], expected_original)
        self.assertAlmostEqual(summary["swapped"]["accuracy_on_parsed"], expected_swapped)
        # A fixed-letter judge scores p in one ordering and 1-p in the other, so the two
        # accuracies always sum to one -- that identity is the invariant worth pinning,
        # and it holds even when the sampled rows happen to split 50/50.
        self.assertAlmostEqual(summary["original"]["accuracy_on_parsed"]
                               + summary["swapped"]["accuracy_on_parsed"], 1.0)
        self.assertAlmostEqual(summary["both_order"]["both_order_accuracy"], 0.0)
        self.assertAlmostEqual(summary["position_consistency"]["position_consistency"], 0.0)
        # Each row remembers its own ordering and the physical answer it named.
        for row in swapped:
            self.assertEqual(row["display_order"],
                             [labels.RESPONSE_2, labels.RESPONSE_1])
            self.assertEqual(row["chosen_response"], labels.RESPONSE_2)
        for row in originals:
            self.assertEqual(row["chosen_response"], labels.RESPONSE_1)

    def test_single_view_run_is_supported_and_reported(self):
        summary, records, _ = self._run(
            "vanilla", lambda messages: "<answer>A</answer>",
            limit=2, views=("original",), progress_every=0)
        self.assertEqual(len(records), 2)
        self.assertEqual(summary["views_run"], ["original"])
        self.assertIsNone(summary["swapped"])
        self.assertNotIn("both_order", summary)
        # A missing ordering is reported, not raised and not silently averaged.
        self.assertFalse(summary["aggregate"]["official_swapped"]["available"])
        self.assertIn("cannot be averaged",
                      summary["aggregate"]["official_swapped"]["reason"])
        self.assertIn("average", summary["aggregate"]["official_original"])

    def test_probability_head_records_targets_end_to_end(self):
        summary, records, _ = self._run(
            "jwc",
            lambda messages: "<think>t</think><prob_rb_better>0.75</prob_rb_better>",
            limit=2, progress_every=0)
        self.assertTrue(all(row["probability"] is not None for row in records))
        self.assertTrue(all(row["probability_target"] in (0, 1) for row in records))
        # p=0.75 is a B decision, mapped back to the physical answer of each ordering.
        self.assertTrue(all(row["probability_decision"] == labels.ANSWER_B
                            for row in records))
        self.assertTrue(all(row["outcome"] == labels.ANSWER_B for row in records))
        self.assertTrue(all(row["correct"] is not None for row in records))
        self.assertTrue(all(row["chosen_response"] is not None for row in records))
        self.assertEqual(summary["calibration"]["count"], 4)
        self.assertAlmostEqual(summary["calibration"]["mean_probability"], 0.75)
        # Targets must flip with the ordering, otherwise calibration is measured in the
        # wrong space and lands at 50%.  Check it per qid: the pair's two targets have to
        # be complements of one another.
        by_qid = {}
        for row in records:
            by_qid.setdefault(row["qid"], {})[row["is_swapped"]] = row["probability_target"]
        self.assertEqual(len(by_qid), 2)
        for qid, targets in by_qid.items():
            self.assertEqual(set(targets), {False, True}, qid)
            self.assertEqual(targets[False] + targets[True], 1, qid)

    def test_a_missing_field_in_the_predictions_file_is_caught(self):
        with _TempDir() as tmp:
            output = os.path.join(tmp, "predictions.jsonl")
            rewardbench.evaluate(method="vanilla",
                                 generate=lambda messages: "<answer>A</answer>",
                                 output_path=output, limit=1, progress_every=0)
            with open(output, encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle if line.strip()]
            del rows[0]["chosen_response"]
            with open(output, "w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")
            with self.assertRaises(ValueError) as ctx:
                rewardbench.load_predictions(output)
            self.assertIn("is missing", str(ctx.exception))

    def test_resume_keeps_completed_rows_and_continues_fixed_source(self):
        with _TempDir() as tmp:
            output = os.path.join(tmp, "predictions.jsonl")
            calls = []
            def generate(messages):
                calls.append(messages)
                return "<answer>A</answer>"
            first = rewardbench.evaluate(
                method="vanilla", generate=generate, output_path=output,
                limit=2, progress_every=0)
            self.assertEqual(first["samples"], 4)
            before = len(calls)
            second = rewardbench.evaluate(
                method="vanilla", generate=generate, output_path=output,
                limit=2, progress_every=0)
            self.assertEqual(second["samples"], 4)
            self.assertTrue(second["resumed"])
            self.assertEqual(second["new_views"], 0)
            self.assertEqual(len(calls), before)
            self.assertEqual(len(rewardbench.load_predictions(output)), 4)
            self.assertTrue(os.path.exists(output + ".subsets.csv"))

    def test_unknown_view_name_is_rejected(self):
        with _TempDir() as tmp:
            with self.assertRaises(ValueError) as ctx:
                rewardbench.evaluate(
                    method="vanilla", generate=lambda messages: "x",
                    output_path=os.path.join(tmp, "p.jsonl"), limit=1,
                    views=("original", "sideways"))
            self.assertIn("unknown view names", str(ctx.exception))

    def test_prompts_come_from_the_training_module(self):
        view = _view("q1", is_swapped=False, ground_truth=labels.ANSWER_A)
        messages = rewardbench.build_prompt_messages("j4r", view)
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("Verdict: [[A>B]]", messages[0]["content"] + messages[1]["content"])
        self.assertIn(view["question"], messages[1]["content"])


if __name__ == "__main__":
    unittest.main()
