"""Tests for the per-rollout log and the degenerate-group diagnostics.

Covers the two prompt requirements that were missing until now:

* Sec. 5.4 -- a multi-rollout group interface that keeps prompt id, completion, parse
  result and reward for every rollout;
* Sec. 5.5 -- explicit handling and recording of all-equal rewards, zero standard
  deviation, and too few valid rollouts, without producing NaN.

These are CPU-only and load no model: they exercise the same functions the trainers call.
"""

from __future__ import annotations

import math
import unittest

from common import prompts
from rewards import correctness, jwc_brier
from rewards import rollout_log as rl
from rewards.base import REASON_INVALID_OUTPUT, REASON_CORRECT, Outcome


class RecordShapeTest(unittest.TestCase):
    def test_a_record_carries_every_field_the_brief_lists(self):
        record = rl.RolloutRecord(
            "hs3-1::original", "<answer>A</answer>", 1.0,
            pair_id="hs3-1", view="original", rollout_index=3,
            reason=REASON_CORRECT, outcome="A", format_ok=True)
        payload = record.as_dict()
        for key in ("prompt_id", "completion", "reward", "reason", "outcome",
                    "pair_id", "view", "rollout_index"):
            self.assertIn(key, payload)
        self.assertEqual(payload["prompt_id"], "hs3-1::original")
        self.assertEqual(payload["rollout_index"], 3)

    def test_a_nan_reward_is_refused(self):
        with self.assertRaises(rl.RolloutLogError) as caught:
            rl.RolloutRecord("p::original", "x", float("nan"))
        self.assertIn("NaN", str(caught.exception))

    def test_an_empty_prompt_id_is_refused(self):
        with self.assertRaises(rl.RolloutLogError):
            rl.RolloutRecord("", "x", 1.0)

    def test_split_sample_id_round_trips(self):
        self.assertEqual(rl.split_sample_id("p::swapped"), ("p", "swapped"))
        self.assertEqual(rl.split_sample_id("p"), ("p", None))


class ValidityTest(unittest.TestCase):
    """An unparsable completion earns 0.0, so validity has to come from the reason."""

    def test_an_invalid_output_is_not_counted_as_valid(self):
        record = rl.RolloutRecord(
            "p::original", "garbage", 0.0, reason=REASON_INVALID_OUTPUT)
        self.assertFalse(record.valid)

    def test_missing_the_verdict_is_not_the_same_as_judging_wrong(self):
        wrong = rl.RolloutRecord("p::original", "<answer>B</answer>", 0.0,
                                 reason="incorrect")
        broken = rl.RolloutRecord("p::original", "garbage", 0.0,
                                  reason=REASON_INVALID_OUTPUT)
        self.assertTrue(wrong.valid)
        self.assertFalse(broken.valid)
        self.assertEqual(wrong.reward, broken.reward)

    def test_without_a_reason_the_reward_is_the_only_evidence(self):
        self.assertTrue(rl.RolloutRecord("p::original", "x", 1.0).valid)
        self.assertFalse(rl.RolloutRecord("p::original", "x", None).valid)


class DiagnosticsTest(unittest.TestCase):
    def _flat_group(self, count=8):
        return rl.records_from_batch(
            ["<answer>A</answer>"] * count, [1.0] * count,
            ["p::original"] * count)

    def test_all_equal_rewards_are_reported_as_flat(self):
        report = rl.group_diagnostics(self._flat_group())
        self.assertEqual(report["flat_groups"], ["p::original"])
        self.assertEqual(report["flat_rollout_ratio"], 1.0)
        self.assertFalse(report["groups"]["p::original"]["advantage_signal"])

    def test_a_flat_group_produces_no_nan(self):
        """The diagnostic must never be the thing that introduces a NaN."""
        report = rl.group_diagnostics(self._flat_group())
        stats = report["groups"]["p::original"]["stats"]
        self.assertEqual(stats["mean"], 1.0)
        self.assertEqual(stats["std"], 0.0)
        self.assertTrue(math.isfinite(stats["std"]))
        for value in (stats["mean"], stats["std"], stats["min"], stats["max"]):
            self.assertFalse(math.isnan(value))

    def test_a_spread_group_is_not_flat(self):
        report = rl.group_diagnostics(rl.records_from_batch(
            ["a", "b", "c", "d"], [1.0, 0.0, 1.0, 0.0], ["p::original"] * 4))
        self.assertEqual(report["flat_groups"], [])
        self.assertTrue(report["groups"]["p::original"]["advantage_signal"])
        self.assertEqual(report["groups"]["p::original"]["stats"]["mean"], 0.5)

    def test_zero_std_is_reported_under_both_names(self):
        entry = rl.group_diagnostics(self._flat_group())["groups"]["p::original"]
        self.assertTrue(entry["flat"])
        self.assertTrue(entry["zero_variance"])

    def test_too_few_valid_rollouts_is_reported(self):
        report = rl.group_diagnostics(rl.records_from_batch(
            ["a", "b", "c"], [1.0, 0.0, 0.0], ["p::original"] * 3,
            reasons=["correct", REASON_INVALID_OUTPUT, REASON_INVALID_OUTPUT]))
        self.assertEqual(report["underfilled_groups"], ["p::original"])
        self.assertEqual(report["groups"]["p::original"]["valid"], 1)
        self.assertEqual(report["groups"]["p::original"]["invalid"], 2)

    def test_a_single_scored_rollout_has_no_standard_deviation(self):
        report = rl.group_diagnostics(rl.records_from_batch(
            ["a"], [1.0], ["p::original"]))
        stats = report["groups"]["p::original"]["stats"]
        self.assertIsNone(stats["std"])
        self.assertEqual(report["underfilled_groups"], ["p::original"])

    def test_a_group_of_eight_broken_parses_is_both_flat_and_underfilled(self):
        """The failure mode the diagnostics exist for: a silent zero-loss plateau."""
        report = rl.group_diagnostics(rl.records_from_batch(
            ["garbage"] * 8, [0.0] * 8, ["p::original"] * 8,
            reasons=[REASON_INVALID_OUTPUT] * 8))
        entry = report["groups"]["p::original"]
        self.assertEqual(report["underfilled_groups"], ["p::original"])
        self.assertTrue(entry["flat"])
        self.assertEqual(report["valid"], 0)
        self.assertEqual(report["invalid"], 8)

    def test_groups_are_kept_separate(self):
        records = rl.records_from_batch(
            ["a", "b", "c", "d"], [1.0, 1.0, 0.0, 0.0],
            ["p::original", "p::original", "q::original", "q::original"])
        report = rl.group_diagnostics(records)
        self.assertEqual(report["group_count"], 2)
        self.assertEqual(report["flat_group_count"], 2)
        self.assertEqual(report["rollouts"], 4)

    def test_an_empty_batch_is_refused(self):
        with self.assertRaises(rl.RolloutLogError):
            rl.group_diagnostics([])


class RecordsFromBatchTest(unittest.TestCase):
    def test_rollout_indices_count_within_a_group(self):
        records = rl.records_from_batch(
            ["a"] * 4, [1.0, 0.0, 1.0, 0.0],
            ["p::original", "p::original", "q::original", "q::original"])
        self.assertEqual([r.rollout_index for r in records], [0, 1, 0, 1])

    def test_a_custom_group_key_splits_by_state(self):
        """J4R normalises within a state, so its diagnostic group is the state."""
        records = rl.records_from_batch(
            ["a"] * 4, [1.0, 1.0, 0.0, 0.0],
            ["p::original", "p::original", "p::swapped", "p::swapped"],
            group_key_fn=lambda sid: sid)
        report = rl.group_diagnostics(records)
        self.assertEqual(report["group_count"], 2)
        self.assertEqual(report["flat_group_count"], 2)

    def test_mismatched_lengths_are_refused(self):
        with self.assertRaises(rl.RolloutLogError):
            rl.records_from_batch(["a"], [1.0, 0.0], ["p::original"])


class LoggedRewardFnTest(unittest.TestCase):
    def test_the_wrapper_returns_exactly_what_it_wrapped(self):
        base = correctness.make_reward_fn("vanilla")
        wrapped = rl.logged_reward_fn(base)
        completions = ["<answer>A</answer>", "<answer>B</answer>"]
        truth = ["A", "A"]
        self.assertEqual(
            wrapped(completions=completions, ground_truth=truth,
                    sample_id=["p::original", "p::original"]),
            base(completions=completions, ground_truth=truth))

    def test_every_rollout_is_logged_with_its_parse_result(self):
        wrapped = rl.logged_reward_fn(correctness.make_reward_fn("vanilla"))
        wrapped(completions=["<answer>A</answer>", "garbage"], ground_truth=["A", "A"],
                sample_id=["p::original", "p::original"])
        records = wrapped.last_rollouts
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0].reason, REASON_CORRECT)
        self.assertEqual(records[1].reason, REASON_INVALID_OUTPUT)
        self.assertEqual(records[0].prompt_id, "p::original")
        self.assertEqual(records[1].completion, "garbage")

    def test_the_diagnostics_are_attached(self):
        wrapped = rl.logged_reward_fn(correctness.make_reward_fn("vanilla"))
        wrapped(completions=["<answer>A</answer>"] * 3, ground_truth=["A"] * 3,
                sample_id=["p::original"] * 3)
        report = wrapped.last_diagnostics
        self.assertIsNotNone(report)
        self.assertEqual(report["groups"]["p::original"]["valid"], 3)

    def test_a_probability_reward_keeps_its_probability_in_the_log(self):
        wrapped = rl.logged_reward_fn(jwc_brier.make_reward_fn("jwc"))
        wrapped(completions=["<think>because</think>"
                             "<prob_rb_better>0.82</prob_rb_better>"],
                ground_truth=["B"], sample_id=["p::original"])
        self.assertAlmostEqual(wrapped.last_rollouts[0].reward, 1 - (0.82 - 1) ** 2)

    def test_j4r_style_negative_rewards_survive_the_log(self):
        """The log must not clip; J4R legitimately scores -0.5."""
        wrapped = rl.logged_reward_fn(correctness.make_reward_fn("vanilla"))
        wrapped(completions=["<answer>B</answer>"], ground_truth=["A"],
                sample_id=["p::original"])
        self.assertEqual(wrapped.last_rollouts[0].reward, 0.0)
        # A hand-built negative reward goes through unchanged.
        records = rl.records_from_batch(
            ["garbage"], [-0.5], ["p::original"],
            reasons=[REASON_INVALID_OUTPUT])
        self.assertEqual(records[0].reward, -0.5)

    def test_a_sample_id_length_mismatch_is_refused(self):
        wrapped = rl.logged_reward_fn(correctness.make_reward_fn("vanilla"))
        with self.assertRaises(rl.RolloutLogError):
            wrapped(completions=["<answer>A</answer>"], ground_truth=["A"],
                    sample_id=["p::original", "q::original"])


class RealPromptShapeTest(unittest.TestCase):
    """The log records the completion verbatim; it must not re-render the prompt."""

    def test_the_logged_completion_is_the_generated_text(self):
        messages = prompts.build_messages(
            "vanilla", question="Q?", response_a="first", response_b="second")
        self.assertEqual(len(messages), 2)
        completion = "<answer>A</answer>"
        wrapped = rl.logged_reward_fn(correctness.make_reward_fn("vanilla"))
        wrapped(completions=[completion], ground_truth=["A"],
                sample_id=["p::original"])
        self.assertEqual(wrapped.last_rollouts[0].completion, completion)


if __name__ == "__main__":
    unittest.main()
