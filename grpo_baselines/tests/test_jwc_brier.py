"""JwC tests: strict probability parsing, Brier arithmetic, reward direction, no clipping.

Paper: arXiv:2510.00263, Sec. 3 Eq. (1): ``R_Brier(tau; y) = 1 - (p - y)^2`` for a parsed
``p in [0, 1]`` and ``0`` otherwise, with ``p = Pr(B better | x)`` and ``y = 1`` when
display B holds the better response.

Two properties matter more than any single number:

* **Reward direction.**  ``y = 1`` means B is better, so a high probability must pay off
  exactly when the label says B.  Getting this backwards trains a confidently wrong judge.
* **No clipping inside Brier.**  The paper's ``eps = 1e-3`` exists to keep ``log`` finite
  in Eq. (2).  Brier is quadratic, so clipping here would silently change the objective --
  the tests below pin the exact arithmetic instead of a tolerance.
"""

from __future__ import annotations

import math
import unittest

from common import parsing
from rewards import jwc_brier
from rewards.base import REASON_INVALID_OUTPUT


def _completion(probability_text: str) -> str:
    return "<think>reasoning</think><prob_rb_better>{0}</prob_rb_better>".format(
        probability_text)


class ProbabilityParserTest(unittest.TestCase):
    """One float, finite, in range, exactly one tag."""

    def test_plain_float_is_parsed(self):
        result = parsing.parse_jwc(_completion("0.82"))
        self.assertTrue(result.ok)
        self.assertTrue(result.format_ok)
        self.assertAlmostEqual(result.probability, 0.82)
        self.assertEqual(result.reasoning, "reasoning")

    def test_boundary_values_are_accepted(self):
        for text, expected in (("0", 0.0), ("1", 1.0), ("0.0", 0.0), ("1.0", 1.0),
                               ("0.000", 0.0), ("1.000", 1.0)):
            result = parsing.parse_jwc(_completion(text))
            self.assertTrue(result.ok, msg=text)
            self.assertEqual(result.probability, expected, msg=text)

    def test_whitespace_inside_the_tag_is_tolerated(self):
        result = parsing.parse_jwc(
            "<think>r</think>< prob_rb_better >  0.5  < / prob_rb_better >")
        self.assertTrue(result.ok)
        self.assertAlmostEqual(result.probability, 0.5)

    def test_out_of_range_values_are_rejected(self):
        for text in ("-0.01", "1.01", "2", "-1", "100"):
            result = parsing.parse_jwc(_completion(text))
            self.assertFalse(result.ok, msg=text)
            self.assertEqual(result.reason, parsing.PROB_OUT_OF_RANGE, msg=text)

    def test_nan_and_infinity_are_rejected(self):
        for text in ("nan", "NaN", "inf", "-inf", "Infinity"):
            result = parsing.parse_jwc(_completion(text))
            self.assertFalse(result.ok, msg=text)
            self.assertEqual(result.reason, parsing.PROB_NOT_FINITE, msg=text)

    def test_a_non_numeric_payload_is_rejected(self):
        for text in ("high", "0.8 or so", "", "  "):
            result = parsing.parse_jwc(_completion(text))
            self.assertFalse(result.ok, msg=text)
            self.assertIn(result.reason,
                          (parsing.PROB_NOT_A_NUMBER, parsing.EMPTY_OUTPUT), msg=text)

    def test_a_missing_tag_is_rejected(self):
        result = parsing.parse_jwc("<think>r</think>I think A is better.")
        self.assertFalse(result.ok)
        self.assertEqual(result.reason, parsing.NO_PROB_SECTION)

    def test_two_probability_tags_are_rejected(self):
        result = parsing.parse_jwc(
            "<think>r</think><prob_rb_better>0.2</prob_rb_better>"
            "<prob_rb_better>0.8</prob_rb_better>")
        self.assertFalse(result.ok)
        self.assertEqual(result.reason, parsing.MULTIPLE_PROB_TAGS)

    def test_an_unclosed_tag_is_rejected(self):
        result = parsing.parse_jwc("<think>r</think><prob_rb_better>0.8")
        self.assertFalse(result.ok)
        self.assertEqual(result.reason, parsing.UNCLOSED_PROB_TAG)

    def test_the_reasoning_block_is_mandatory(self):
        # The paper's "Verbalized Confidence w/ CoT" prompt asks for the reasoning block
        # first, so a bare float is a format violation, not a valid answer.
        result = parsing.parse_jwc("<prob_rb_better>0.8</prob_rb_better>")
        self.assertFalse(result.ok)
        self.assertEqual(result.reason, parsing.MISSING_THINK)


class BrierArithmeticTest(unittest.TestCase):
    """``1 - (p - y)^2`` with no clipping and no epsilon fiddling."""

    def test_perfectly_correct_and_confident_scores_one(self):
        self.assertEqual(jwc_brier.brier_reward(1.0, 1), 1.0)
        self.assertEqual(jwc_brier.brier_reward(0.0, 0), 1.0)

    def test_perfectly_wrong_and_confident_scores_zero(self):
        self.assertEqual(jwc_brier.brier_reward(0.0, 1), 0.0)
        self.assertEqual(jwc_brier.brier_reward(1.0, 0), 0.0)

    def test_neutral_probability_scores_the_same_either_way(self):
        self.assertEqual(jwc_brier.brier_reward(0.5, 0), jwc_brier.brier_reward(0.5, 1))
        self.assertAlmostEqual(jwc_brier.brier_reward(0.5, 1), 0.75)

    def test_the_reward_is_quadratic_in_the_error(self):
        # p - y = 0.1 and 0.2 must differ by more than a linear gap: 0.99 vs 0.96.
        self.assertAlmostEqual(jwc_brier.brier_reward(0.9, 1), 0.99)
        self.assertAlmostEqual(jwc_brier.brier_reward(0.8, 1), 0.96)
        self.assertAlmostEqual(jwc_brier.brier_reward(0.1, 1), 0.19)

    def test_the_reward_is_symmetric_around_the_target(self):
        for delta in (0.1, 0.25, 0.4):
            self.assertAlmostEqual(jwc_brier.brier_reward(1.0 - delta, 1),
                                   jwc_brier.brier_reward(0.0 + delta, 0))

    def test_malformed_inputs_raise(self):
        with self.assertRaises(ValueError):
            jwc_brier.brier_reward(float("nan"), 1)
        with self.assertRaises(ValueError):
            jwc_brier.brier_reward(1.5, 1)
        with self.assertRaises(ValueError):
            jwc_brier.brier_reward(0.5, 2)

    def test_brier_does_not_clip_extreme_probabilities(self):
        # A clipped implementation would return 1 - (1 - 1e-3 - 1)^2 = 0.999999 instead of
        # exactly 0.0 for a confidently wrong answer.
        self.assertEqual(jwc_brier.brier_reward(1.0, 0), 0.0)
        self.assertEqual(jwc_brier.brier_reward(0.0, 1), 0.0)
        self.assertNotAlmostEqual(jwc_brier.brier_reward(1.0, 0), 1e-6)

    def test_the_log_reward_is_the_one_that_clips(self):
        # Eq. (2) clips to [eps, 1-eps]; that is why the two rewards differ at the extremes.
        epsilon = jwc_brier.DEFAULT_EPSILON
        self.assertAlmostEqual(jwc_brier.log_reward(0.0, 1), math.log(epsilon))
        self.assertAlmostEqual(jwc_brier.log_reward(1.0, 0), math.log(epsilon))
        self.assertAlmostEqual(jwc_brier.log_reward(1.0, 1), math.log(1.0 - epsilon))
        self.assertLess(jwc_brier.log_reward(0.0, 1), jwc_brier.brier_reward(0.0, 1))
        with self.assertRaises(ValueError):
            jwc_brier.log_reward(0.5, 1, epsilon=0.0)
        with self.assertRaises(ValueError):
            jwc_brier.log_reward(0.5, 1, epsilon=0.5)


class RewardDirectionTest(unittest.TestCase):
    """A high probability must pay off when, and only when, display B is the better one."""

    def test_high_probability_scores_well_when_b_is_better(self):
        outcome = jwc_brier.score_text(_completion("0.95"), "B")
        self.assertAlmostEqual(outcome.reward, 0.9975)
        self.assertEqual(outcome.detail["target"], 1)
        self.assertTrue(outcome.detail["correct"])

    def test_high_probability_scores_poorly_when_a_is_better(self):
        outcome = jwc_brier.score_text(_completion("0.95"), "A")
        self.assertAlmostEqual(outcome.reward, 0.0975)
        self.assertEqual(outcome.detail["target"], 0)
        self.assertFalse(outcome.detail["correct"])

    def test_low_probability_scores_well_when_a_is_better(self):
        outcome = jwc_brier.score_text(_completion("0.05"), "A")
        self.assertAlmostEqual(outcome.reward, 0.9975)
        self.assertTrue(outcome.detail["correct"])

    def test_reward_is_monotone_in_confidence_along_the_correct_direction(self):
        values = [jwc_brier.score_text(_completion(str(p)), "B").reward
                  for p in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)]
        self.assertEqual(values, sorted(values))
        self.assertLess(values[0], values[-1])

    def test_a_confidently_wrong_answer_is_worse_than_a_hedged_one(self):
        hedged = jwc_brier.score_text(_completion("0.5"), "A").reward
        wrong = jwc_brier.score_text(_completion("1.0"), "A").reward
        self.assertGreater(hedged, wrong)
        self.assertEqual(wrong, 0.0)

    def test_the_decision_boundary_is_strictly_above_one_half(self):
        # p = 0.5 is a tie in confidence, and the shipped rule resolves it to A.
        self.assertEqual(jwc_brier.score_text(_completion("0.5"), "A").detail["decision"], "A")
        self.assertEqual(jwc_brier.score_text(_completion("0.5"), "B").detail["decision"], "A")
        self.assertEqual(jwc_brier.score_text(_completion("0.51"), "B").detail["decision"], "B")


class InvalidOutputTest(unittest.TestCase):
    """Unparsable output earns exactly 0, with the parser's reason preserved."""

    def test_unparsable_earns_zero_not_the_neutral_reward(self):
        outcome = jwc_brier.score_text("<think>r</think>no float", "B")
        self.assertEqual(outcome.reward, 0.0)
        self.assertEqual(outcome.reason, REASON_INVALID_OUTPUT)
        # 0.75 is what a neutral 0.5 would earn; silence must not be that profitable.
        self.assertNotAlmostEqual(outcome.reward, 0.75)

    def test_the_parser_reason_survives(self):
        outcome = jwc_brier.score_text(_completion("1.5"), "B")
        self.assertEqual(outcome.reason, REASON_INVALID_OUTPUT)
        self.assertEqual(outcome.detail["parse_reason"], parsing.PROB_OUT_OF_RANGE)

    def test_the_target_is_still_recorded_for_invalid_rows(self):
        outcome = jwc_brier.score_text(_completion("nan"), "B")
        self.assertEqual(outcome.detail["target"], 1)
        self.assertIsNone(outcome.detail.get("probability"))

    def test_an_empty_completion_earns_zero(self):
        self.assertEqual(jwc_brier.score_text("", "A").reward, 0.0)


class RewardFunctionTest(unittest.TestCase):
    """The TRL-facing adapter: one reward per row, never NaN, summarisable."""

    def test_rewards_are_returned_per_completion(self):
        fn = jwc_brier.make_reward_fn("jwc")
        rewards = fn(
            completions=[_completion("0.9"), _completion("0.1"), "<think>r</think>bad"],
            ground_truth=["B", "B", "B"],
        )
        self.assertEqual(len(rewards), 3)
        self.assertAlmostEqual(rewards[0], 0.99)
        self.assertAlmostEqual(rewards[1], 0.19)
        self.assertEqual(rewards[2], 0.0)

    def test_no_reward_is_nan(self):
        fn = jwc_brier.make_reward_fn("jwc")
        rewards = fn(completions=[_completion("nan"), ""], ground_truth=["A", "B"])
        self.assertFalse(any(value != value for value in rewards))

    def test_a_length_mismatch_is_rejected(self):
        fn = jwc_brier.make_reward_fn("jwc")
        with self.assertRaises(ValueError):
            fn(completions=[_completion("0.5")], ground_truth=["A", "B"])

    def test_a_single_ground_truth_string_is_accepted(self):
        fn = jwc_brier.make_reward_fn("jwc")
        self.assertAlmostEqual(
            fn(completions=[_completion("1.0")], ground_truth="B")[0], 1.0)

    def test_conversational_completions_are_unwrapped(self):
        fn = jwc_brier.make_reward_fn("jwc")
        rewards = fn(
            completions=[[{"role": "assistant", "content": _completion("1.0")}]],
            ground_truth=["B"],
        )
        self.assertAlmostEqual(rewards[0], 1.0)

    def test_summary_reports_brier_and_accuracy_separately(self):
        items = [
            jwc_brier.score_text(_completion("0.9"), "B"),
            jwc_brier.score_text(_completion("0.6"), "A"),
            jwc_brier.score_text(_completion("0.2"), "B"),
            jwc_brier.score_text("<think>r</think>bad", "A"),
        ]
        stats = jwc_brier.summarise_probabilities(items)
        self.assertEqual(stats["count"], 4)
        self.assertEqual(stats["valid"], 3)
        self.assertEqual(stats["invalid"], 1)
        # ``brier`` is the *loss* (p - y)^2, matching evaluation.metrics.brier_score, so
        # lower is better and the reward is its complement:
        #   p=0.9 vs y=1 -> 0.01    p=0.6 vs y=0 -> 0.36    p=0.2 vs y=1 -> 0.64
        expected_brier = (0.01 + 0.36 + 0.64) / 3
        self.assertAlmostEqual(stats["brier"], expected_brier)
        # ``brier`` averages over parsable rows only, while ``mean_reward`` averages over
        # the whole batch with the invalid row contributing 0 -- so the two differ by the
        # invalid rate, which is exactly the "silence earns nothing" rule.
        self.assertAlmostEqual(stats["mean_reward"],
                               (1.0 - expected_brier) * stats["valid"] / stats["count"])
        self.assertLess(stats["mean_reward"], 1.0 - expected_brier)
        # Decisions: B (right), B (wrong), A (wrong) -> 1/3.
        self.assertAlmostEqual(stats["decision_accuracy"], 1 / 3)
        self.assertAlmostEqual(stats["mean_probability"], (0.9 + 0.6 + 0.2) / 3)

    def test_summary_handles_an_all_invalid_batch(self):
        items = [jwc_brier.score_text("bad", "A")]
        stats = jwc_brier.summarise_probabilities(items)
        self.assertEqual(stats["invalid"], 1)
        self.assertEqual(stats["valid"], 0)
        self.assertEqual(stats["invalid_rate"], 1.0)
        # Nothing parsed, so the probability-derived fields stay empty instead of
        # pretending a 0.5 fallback was observed.
        self.assertIsNone(stats["brier"])
        self.assertIsNone(stats["mean_probability"])
        self.assertIsNone(stats["decision_accuracy"])
        self.assertEqual(stats["mean_reward"], 0.0)

    def test_summary_rejects_an_empty_batch(self):
        with self.assertRaises(ValueError):
            jwc_brier.summarise_probabilities([])


class GroundTruthSpaceTest(unittest.TestCase):
    """``y`` is defined on the display slot, so a swapped view flips it."""

    def test_probability_target_flips_with_the_view(self):
        from common import labels

        self.assertEqual(labels.probability_target("B"), 1)
        self.assertEqual(labels.probability_target("A"), 0)

    def test_the_same_probability_is_good_in_one_order_and_bad_in_the_other(self):
        # p = Pr(B better). The judge says B confidently; that is right only where the
        # swapped view puts the better response in slot B.
        original = jwc_brier.score_text(_completion("1.0"), "A")
        swapped = jwc_brier.score_text(_completion("1.0"), "B")
        self.assertEqual(original.reward, 0.0)
        self.assertEqual(swapped.reward, 1.0)


if __name__ == "__main__":
    unittest.main()
