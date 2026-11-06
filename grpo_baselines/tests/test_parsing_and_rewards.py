"""Parser and reward tests for the vanilla baseline (plus shared parser contracts).

The parsers are the only place where a model's free-form text becomes a number, so they
are tested adversarially: pseudo-matches, duplicated tags, unclosed tags and letters
hidden inside a verdict string must all fail rather than be truncated into an answer.

The last class pins the *upstream* arithmetic this project delegates to TRL.  It reads the
pinned source file as text instead of importing it (``trl`` is not installed in the
read-only environment and importing it would require ``datasets``).
"""

from __future__ import annotations

import math
import os
import unittest

from common import labels, parsing, paths
from rewards import correctness
from rewards.base import (
    REASON_CORRECT,
    REASON_INCORRECT,
    REASON_INVALID_OUTPUT,
)


class VanillaParserTest(unittest.TestCase):
    """``<answer>A</answer>`` with a lenient inner payload and a strict outer shape."""

    def test_plain_letter(self):
        result = parsing.parse_vanilla("<answer>A</answer>")
        self.assertTrue(result.ok)
        self.assertTrue(result.format_ok)
        self.assertEqual(result.outcome, "A")
        self.assertEqual(result.reason, parsing.OK)

    def test_whitespace_and_newlines_are_tolerated(self):
        for text in (
            "<answer>\n  A\n</answer>",
            "< answer >A< / answer >",
            "<ANSWER>a</ANSWER>",
            "some reasoning\n<answer>  b  </answer>",
        ):
            result = parsing.parse_vanilla(text)
            self.assertTrue(result.ok, msg=text)
            self.assertIn(result.outcome, ("A", "B"))

    def test_stacked_brackets_are_tolerated(self):
        self.assertEqual(parsing.parse_vanilla("<answer>[[A]]</answer>").outcome, "A")
        self.assertEqual(parsing.parse_vanilla("<answer>[b]</answer>").outcome, "B")

    def test_pseudo_match_without_a_tag_is_rejected(self):
        result = parsing.parse_vanilla("The better answer is A.")
        self.assertFalse(result.ok)
        self.assertIsNone(result.outcome)
        self.assertEqual(result.reason, parsing.NO_ANSWER_TAG)

    def test_verdict_style_payload_is_not_truncated_to_a_letter(self):
        result = parsing.parse_vanilla("<answer>A>B</answer>")
        self.assertFalse(result.ok)
        self.assertEqual(result.reason, parsing.INVALID_ANSWER_LETTER)

    def test_sentence_payload_is_rejected(self):
        result = parsing.parse_vanilla("<answer>A is better</answer>")
        self.assertFalse(result.ok)
        self.assertEqual(result.reason, parsing.INVALID_ANSWER_LETTER)

    def test_two_tags_are_rejected_even_when_they_agree(self):
        result = parsing.parse_vanilla("<answer>A</answer><answer>A</answer>")
        self.assertFalse(result.ok)
        self.assertEqual(result.reason, parsing.MULTIPLE_ANSWER_TAGS)

    def test_unclosed_tag_is_rejected(self):
        result = parsing.parse_vanilla("<answer>A")
        self.assertFalse(result.ok)
        self.assertEqual(result.reason, parsing.UNCLOSED_ANSWER_TAG)

    def test_empty_output(self):
        for text in ("", "   \n\t "):
            result = parsing.parse_vanilla(text)
            self.assertFalse(result.ok)
            self.assertEqual(result.reason, parsing.EMPTY_OUTPUT)

    def test_non_string_is_a_type_error(self):
        with self.assertRaises(TypeError):
            parsing.parse_vanilla(None)

    def test_registry_rejects_unknown_formats(self):
        with self.assertRaises(ValueError) as ctx:
            parsing.get_parser("j2")
        self.assertIn("unknown output_format", str(ctx.exception))

    def test_every_format_in_the_config_schema_has_a_parser(self):
        for name in parsing.PARSERS:
            self.assertTrue(callable(parsing.get_parser(name)))


class VanillaRewardTest(unittest.TestCase):
    """Correctness reward: 1.0 correct, 0.0 wrong, 0.0 unparsable, with a reason code."""

    def test_correct_answer_scores_one(self):
        outcome = correctness.score_text("<answer>A</answer>", "A")
        self.assertEqual(outcome.reward, 1.0)
        self.assertEqual(outcome.reason, REASON_CORRECT)

    def test_wrong_answer_scores_zero(self):
        outcome = correctness.score_text("<answer>B</answer>", "A")
        self.assertEqual(outcome.reward, 0.0)
        self.assertEqual(outcome.reason, REASON_INCORRECT)

    def test_unparsable_scores_zero_with_a_stable_reason(self):
        outcome = correctness.score_text("I cannot decide.", "A")
        self.assertEqual(outcome.reward, 0.0)
        self.assertEqual(outcome.reason, REASON_INVALID_OUTPUT)
        self.assertFalse(outcome.valid)

    def test_invalid_reason_preserves_the_parser_code(self):
        outcome = correctness.score_text("<answer>A>B</answer>", "A")
        self.assertEqual(outcome.detail["parse_reason"], parsing.INVALID_ANSWER_LETTER)

    def test_outcome_rejects_nan(self):
        from rewards.base import Outcome

        with self.assertRaises(ValueError):
            Outcome(float("nan"), "scored", {})

    def test_reward_fn_matches_the_trl_signature(self):
        fn = correctness.make_reward_fn("vanilla")
        rewards = fn(
            prompts=["p1", "p2", "p3"],
            completions=[
                "<answer>A</answer>",
                "<answer>A</answer>",
                "nothing here",
            ],
            completion_ids=None,
            ground_truth=["A", "B", "A"],
        )
        self.assertEqual(rewards, [1.0, 0.0, 0.0])

    def test_reward_fn_accepts_a_single_ground_truth_string(self):
        fn = correctness.make_reward_fn("vanilla")
        self.assertEqual(
            fn(completions=["<answer>B</answer>"], ground_truth="B"), [1.0])

    def test_reward_fn_rejects_a_length_mismatch(self):
        fn = correctness.make_reward_fn("vanilla")
        with self.assertRaises(ValueError) as ctx:
            fn(completions=["<answer>A</answer>"], ground_truth=["A", "B"])
        self.assertIn("completions", str(ctx.exception))

    def test_reward_fn_accepts_a_conversational_completion(self):
        fn = correctness.make_reward_fn("vanilla")
        rewards = fn(
            completions=[[{"role": "assistant", "content": "<answer>B</answer>"}]],
            ground_truth=["B"],
        )
        self.assertEqual(rewards, [1.0])

    def test_summary_separates_accuracy_from_invalid_rate(self):
        outcomes = [
            correctness.score_text("<answer>A</answer>", "A"),   # correct
            correctness.score_text("<answer>B</answer>", "A"),   # wrong
            correctness.score_text("no tag", "A"),               # invalid
            correctness.score_text("<answer>B</answer>", "B"),   # correct
        ]
        stats = correctness.summary_stats(outcomes)
        self.assertEqual(stats["count"], 4)
        self.assertEqual(stats["valid"], 3)
        self.assertEqual(stats["invalid"], 1)
        self.assertAlmostEqual(stats["invalid_rate"], 0.25)
        # Accuracy on what parsed vs accuracy when silence counts as wrong: a method
        # cannot buy accuracy by refusing to answer.
        self.assertAlmostEqual(stats["accuracy_on_valid"], 2 / 3)
        self.assertAlmostEqual(stats["accuracy_all_treated_wrong"], 0.5)

    def test_summary_rejects_an_empty_batch(self):
        with self.assertRaises(ValueError):
            correctness.summary_stats([])


class GroupAdvantageContractTest(unittest.TestCase):
    """The arithmetic this project delegates to TRL, pinned at the commit it depends on.

    ``vanilla``/``j1``/``jwc`` do not implement group normalisation -- TRL's
    ``GRPOTrainer`` does, and this project configures it with ``scale_rewards="group"``
    and ``loss_type="grpo"``.  The test therefore checks the *vendored source* for the
    three expressions the configuration relies on, and evaluates the same formula to show
    what it does to a degenerate (all-equal) reward vector.
    """

    SOURCE = os.path.join(str(paths.EXTERNAL_DIR), "trl", "trl", "trainer",
                          "grpo_trainer.py")

    def setUp(self):
        if not os.path.exists(self.SOURCE):
            self.skipTest("vendored TRL source is not present")

    def _source(self):
        with open(self.SOURCE, encoding="utf-8") as handle:
            return handle.read()

    def test_upstream_still_centres_and_scales_by_group(self):
        text = self._source()
        self.assertIn("advantages = rewards - mean_grouped_rewards", text)
        self.assertIn("advantages = advantages / (std_rewards + 1e-4)", text)

    def test_upstream_still_neutralises_nan(self):
        self.assertIn("torch.nan_to_num(advantages, nan=0.0)", self._source())

    def test_group_statistics_are_computed_per_prompt_group(self):
        self.assertIn("rewards.view(-1, num_generations)", self._source())

    def test_the_formula_gives_zero_for_a_degenerate_group(self):
        # This mirrors TRL's expression on a group whose rewards are all identical: the
        # numerator is 0 and the denominator is 0 + 1e-4, so the advantage is exactly 0.0
        # rather than a division by zero.
        def advantages(rewards, num_generations):
            groups = [rewards[i:i + num_generations]
                      for i in range(0, len(rewards), num_generations)]
            means, stds = [], []
            for group in groups:
                mean = sum(group) / len(group)
                var = sum((r - mean) ** 2 for r in group) / len(group)
                means.extend([mean] * len(group))
                stds.extend([math.sqrt(var)] * len(group))
            return [(r - m) / (s + 1e-4) for r, m, s in zip(rewards, means, stds)]

        flat = advantages([1.0] * 8, 8)
        self.assertEqual(set(flat), {0.0})
        self.assertFalse(any(math.isnan(value) for value in flat))

        spread = advantages([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0], 8)
        self.assertTrue(all(math.isfinite(value) for value in spread))
        self.assertAlmostEqual(sum(spread), 0.0, places=6)
        self.assertGreater(max(spread), 0.0)
        self.assertLess(min(spread), 0.0)


class ParserCrossFormatTest(unittest.TestCase):
    """A completion written for one method must not be silently accepted by another."""

    SAMPLES = {
        "vanilla": "<answer>A</answer>",
        "j1": "<think>compare</think><answer>[[A]]</answer>",
        "j4r": "Explanation: A wins.\nVerdict: [[A>B]]",
        # The JwC paper's "Verbalized Confidence w/ CoT" prompt (Appendix, prompt 2) asks
        # for the reasoning block and *then* the float, so the sample has both.
        "jwc": "<think>A is clearer.</think><prob_rb_better>0.82</prob_rb_better>",
    }

    def test_each_parser_accepts_its_own_layout(self):
        for method, text in self.SAMPLES.items():
            result = parsing.parse(method, text)
            self.assertTrue(result.ok, msg="{0}: {1}".format(method, result.reason))
            self.assertTrue(result.format_ok, msg=method)

    def test_a_missing_think_block_is_rejected_where_the_prompt_demands_it(self):
        # J1's seed prompt and the JwC CoT prompt both make the reasoning block part of
        # the contract, so the parse fails before the answer/probability is even read.
        bare_answer = "<answer>[[A]]</answer>"
        bare_probability = "<prob_rb_better>0.82</prob_rb_better>"
        self.assertEqual(parsing.parse("j1", bare_answer).reason, parsing.MISSING_THINK)
        self.assertEqual(parsing.parse("jwc", bare_probability).reason, parsing.MISSING_THINK)
        # ... and the format gate runs first, so a foreign completion reports the missing
        # think block rather than a missing probability.
        self.assertEqual(parsing.parse("jwc", self.SAMPLES["vanilla"]).reason,
                         parsing.MISSING_THINK)

    def test_j4r_requires_its_verdict_keyword(self):
        self.assertEqual(parsing.parse("j4r", self.SAMPLES["j1"]).reason,
                         parsing.NO_VERDICT_KEYWORD)
        self.assertEqual(parsing.parse("j4r", self.SAMPLES["vanilla"]).reason,
                         parsing.NO_VERDICT_KEYWORD)

    def test_vanilla_deliberately_accepts_a_bracketed_letter(self):
        # Documented leniency, not an accident: ``[[A]]`` is an unambiguous statement that
        # A is better, and vanilla's reward is only "was the verdict right".  The strict
        # part of vanilla is the *outer* tag, not the inner payload.
        result = parsing.parse("vanilla", self.SAMPLES["j1"])
        self.assertTrue(result.ok)
        self.assertEqual(result.outcome, "A")
        # A verdict-style or explanation-only completion still fails: there is no
        # <answer> tag to anchor on.
        self.assertEqual(parsing.parse("vanilla", self.SAMPLES["j4r"]).reason,
                         parsing.NO_ANSWER_TAG)


if __name__ == "__main__":
    unittest.main()
