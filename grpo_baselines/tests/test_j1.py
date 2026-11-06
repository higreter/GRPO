"""J1 tests: the linked (original, swapped) pair, and the both-correct consistency term.

The J1 paper (arXiv:2505.10320, Sec. 2.2) defines two terms only -- verdict correctness
and verdict consistency -- where consistency is ``+1`` **only when both orders are judged
correctly**.  That definition is the whole reason the method needs two states, so these
tests are written to fail if the implementation ever degrades to "the two letters agree",
which is a completely different (and much easier) property.

The pair identity is tracked through the ``"<pair_id>::<view>"`` sample id that TRL passes
back to the reward function as an ordinary dataset column.
"""

from __future__ import annotations

import unittest

from common import parsing
from data import pair_views
from rewards import j1


def _original(letter: str) -> str:
    return "<think>reasoning</think><answer>[[{0}]]</answer>".format(letter)


def _parse(text: str) -> parsing.ParseResult:
    return parsing.parse_j1(text)


class PairScoringTest(unittest.TestCase):
    """One pair, two rows, three possible pair totals."""

    def test_both_orders_correct_earns_the_consistency_bonus(self):
        result = j1.score_texts(_original("A"), _original("B"), "A", "B")
        self.assertTrue(result["original_correct"])
        self.assertTrue(result["swapped_correct"])
        self.assertTrue(result["both_correct"])
        self.assertTrue(result["same_response_chosen"])
        # 1 (correct) + 1 (both correct) on each row.
        self.assertEqual(result["original_reward"], 2.0)
        self.assertEqual(result["swapped_reward"], 2.0)
        self.assertEqual(result["pair_reward"], 4.0)

    def test_one_order_wrong_keeps_only_the_correctness_point(self):
        result = j1.score_texts(_original("A"), _original("A"), "A", "B")
        self.assertTrue(result["original_correct"])
        self.assertFalse(result["swapped_correct"])
        self.assertFalse(result["both_correct"])
        self.assertEqual(result["original_reward"], 1.0)
        self.assertEqual(result["swapped_reward"], 0.0)
        self.assertEqual(result["pair_reward"], 1.0)

    def test_both_orders_wrong_scores_nothing(self):
        result = j1.score_texts(_original("B"), _original("A"), "A", "B")
        self.assertFalse(result["both_correct"])
        self.assertEqual(result["pair_reward"], 0.0)

    def test_unparsable_row_is_wrong_and_costs_the_bonus(self):
        result = j1.score_texts(
            _original("A"), "<think>reasoning</think>no verdict", "A", "B")
        self.assertTrue(result["original_correct"])
        self.assertFalse(result["swapped_correct"])
        self.assertFalse(result["both_correct"])
        self.assertEqual(result["swapped_reason"], parsing.NO_ANSWER_TAG)
        self.assertEqual(result["swapped_reward"], 0.0)

    def test_a_missing_think_block_makes_the_row_wrong(self):
        # The J1 prompt makes <think> mandatory, so a bare verdict cannot earn the point
        # even when the letter happens to be right.
        result = j1.score_texts("<answer>[[A]]</answer>", _original("B"), "A", "B")
        self.assertFalse(result["original_correct"])
        self.assertEqual(result["original_reason"], parsing.MISSING_THINK)
        self.assertFalse(result["both_correct"])

    def test_the_two_ground_truths_must_be_opposite(self):
        with self.assertRaises(ValueError) as ctx:
            j1.score_texts(_original("A"), _original("B"), "A", "A")
        self.assertIn("opposite ground truths", str(ctx.exception))

    def test_malformed_ground_truth_is_rejected(self):
        with self.assertRaises(ValueError):
            j1.score_texts(_original("A"), _original("B"), "A", "C")


class PositionBiasTest(unittest.TestCase):
    """The property that "same letter in both orders" must NOT be mistaken for consistency.

    A position-biased judge always answers A.  It is right in the original order and wrong
    in the swapped order, so consistency is 0 even though its two letters are identical.
    """

    def test_always_answering_a_is_not_consistent(self):
        result = j1.score_texts(_original("A"), _original("A"), "A", "B")
        self.assertFalse(result["both_correct"])
        self.assertFalse(result["same_response_chosen"])

    def test_a_position_neutral_judge_is_consistent_with_different_letters(self):
        result = j1.score_texts(_original("A"), _original("B"), "A", "B")
        self.assertTrue(result["both_correct"])
        self.assertTrue(result["same_response_chosen"])
        # The letters differ, the physical choice does not -- this is the case that a
        # naive "letters must match" implementation would wrongly mark inconsistent.
        self.assertNotEqual(result["original_outcome"], result["swapped_outcome"])

    def test_consistency_tracks_physical_identity_not_the_letter(self):
        # Right letter, wrong reason: both orders correct => consistent.
        consistent = j1.score_texts(_original("A"), _original("B"), "A", "B")
        # Swapped answer in the swapped view means the judge followed position: wrong.
        biased = j1.score_texts(_original("A"), _original("A"), "A", "B")
        self.assertTrue(consistent["same_response_chosen"])
        self.assertFalse(biased["same_response_chosen"])


class RewardFunctionTest(unittest.TestCase):
    """The TRL-facing reward function: grouping by ``sample_id``, no NaN, stable stats."""

    def _rows(self):
        return [
            # pair 1: both correct
            {"sample_id": "p1::original", "gt": "A", "text": _original("A")},
            {"sample_id": "p1::swapped", "gt": "B", "text": _original("B")},
            # pair 2: the original order is judged wrong, the swapped order rightly
            {"sample_id": "p2::original", "gt": "B", "text": _original("A")},
            {"sample_id": "p2::swapped", "gt": "A", "text": _original("A")},
        ]

    def test_rewards_are_per_row_and_the_pair_bonus_is_broadcast(self):
        rows = self._rows()
        fn = j1.make_reward_fn("j1")
        rewards = fn(
            completions=[row["text"] for row in rows],
            ground_truth=[row["gt"] for row in rows],
            sample_id=[row["sample_id"] for row in rows],
        )
        self.assertEqual(len(rewards), 4)
        self.assertEqual(rewards[0], 2.0)   # p1 original: correct + both
        self.assertEqual(rewards[1], 2.0)   # p1 swapped:  correct + both
        self.assertEqual(rewards[2], 0.0)   # p2 original: wrong, no bonus
        self.assertEqual(rewards[3], 1.0)   # p2 swapped:  correct, no bonus
        self.assertEqual(fn.last_stats["paired_groups"], 2)
        self.assertEqual(fn.last_stats["unpaired_groups"], 0)
        self.assertEqual(fn.last_stats["rows"], 4)

    def test_rewards_never_contain_nan(self):
        rows = self._rows()
        rows[3]["text"] = "garbage"
        fn = j1.make_reward_fn("j1")
        rewards = fn(
            completions=[row["text"] for row in rows],
            ground_truth=[row["gt"] for row in rows],
            sample_id=[row["sample_id"] for row in rows],
        )
        self.assertTrue(all(isinstance(value, float) for value in rewards))
        self.assertFalse(any(value != value for value in rewards))

    def test_a_row_without_a_counterpart_degrades_to_correctness_only(self):
        # This is the failure mode the trainer-side guard exists for: the reward is still
        # a valid number, so nothing crashes -- it silently loses the consistency term.
        fn = j1.make_reward_fn("j1")
        rewards = fn(
            completions=[_original("A")],
            ground_truth=["A"],
            sample_id=["lonely::original"],
        )
        self.assertEqual(rewards, [1.0])
        self.assertEqual(fn.last_stats["unpaired_groups"], 1)
        self.assertEqual(fn.last_stats["paired_groups"], 0)

    def test_sample_id_is_required(self):
        fn = j1.make_reward_fn("j1")
        with self.assertRaises(ValueError) as ctx:
            fn(completions=[_original("A")], ground_truth=["A"])
        self.assertIn("sample_id", str(ctx.exception))

    def test_a_sample_id_without_a_view_is_rejected(self):
        fn = j1.make_reward_fn("j1")
        with self.assertRaises(ValueError) as ctx:
            fn(completions=[_original("A")], ground_truth=["A"], sample_id=["p1"])
        self.assertIn("::", str(ctx.exception))

    def test_grouping_survives_rows_arriving_out_of_order(self):
        # The sampler groups the two states adjacently, but the reward function must not
        # depend on that: it groups by identity, not by position in the batch.
        fn = j1.make_reward_fn("j1")
        rewards = fn(
            completions=[_original("B"), _original("A")],
            ground_truth=["B", "A"],
            sample_id=["p1::swapped", "p1::original"],
        )
        self.assertEqual(rewards, [2.0, 2.0])
        self.assertEqual(fn.last_stats["paired_groups"], 1)

    def test_a_duplicated_view_is_rejected_instead_of_scored_as_zero(self):
        # A duplicate used to overwrite the earlier row and leave it at the default 0.0 --
        # a reward that was never computed.  The sampler cannot produce this, so treat it
        # as a malformed batch rather than training on a silent wrong signal.
        fn = j1.make_reward_fn("j1")
        with self.assertRaises(ValueError) as ctx:
            fn(
                completions=[_original("A"), _original("B"), _original("A")],
                ground_truth=["A", "B", "A"],
                sample_id=["p1::original", "p1::swapped", "p1::original"],
            )
        self.assertIn("duplicate view", str(ctx.exception))

    def test_three_pairs_in_one_batch_are_scored_independently(self):
        fn = j1.make_reward_fn("j1")
        rewards = fn(
            completions=[_original("A"), _original("B"),
                         _original("A"), _original("A"),
                         _original("B"), _original("B")],
            ground_truth=["A", "B", "B", "A", "A", "B"],
            sample_id=["p1::original", "p1::swapped",
                       "p2::original", "p2::swapped",
                       "p3::original", "p3::swapped"],
        )
        self.assertEqual(fn.last_stats["paired_groups"], 3)
        self.assertEqual(fn.last_stats["unpaired_groups"], 0)
        # p1 both right; p2 original wrong, swapped right; p3 original wrong, swapped right.
        self.assertEqual(rewards, [2.0, 2.0, 0.0, 1.0, 0.0, 1.0])


class RealPairIdentityTest(unittest.TestCase):
    """Cross-check the reward against the pair-view layer's notion of identity."""

    def setUp(self):
        self.views = pair_views.build_views(
            pair_id="hs3-1",
            source_row_index=1,
            question="q",
            answer_1="first candidate",
            answer_2="second candidate",
            better_response="answer1",
            source_dataset="helpsteer3_train_clean",
            two_states=True,
        )
        self.original, self.swapped = self.views

    def test_the_swapped_view_really_flips_the_label(self):
        self.assertEqual(self.original.ground_truth, "A")
        self.assertEqual(self.swapped.ground_truth, "B")
        self.assertNotEqual(self.original.answer_a, self.swapped.answer_a)

    def test_correct_verdicts_in_both_orders_are_the_same_physical_choice(self):
        result = j1.score_texts(
            _original(self.original.ground_truth),
            _original(self.swapped.ground_truth),
            self.original.ground_truth,
            self.swapped.ground_truth,
        )
        self.assertTrue(result["both_correct"])
        self.assertEqual(
            pair_views.identity_of(self.original, result["original_outcome"]),
            pair_views.identity_of(self.swapped, result["swapped_outcome"]),
        )

    def test_identical_display_letters_across_views_are_different_answers(self):
        result = j1.score_texts(
            _original("A"), _original("A"),
            self.original.ground_truth, self.swapped.ground_truth,
        )
        self.assertFalse(result["both_correct"])
        self.assertNotEqual(
            pair_views.identity_of(self.original, result["original_outcome"]),
            pair_views.identity_of(self.swapped, result["swapped_outcome"]),
        )


if __name__ == "__main__":
    unittest.main()
