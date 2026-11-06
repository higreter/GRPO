"""J4R advantage tests: EIS grouping, hand-computed advantages, zero variance, state swap.

The J4R paper (arXiv:2505.13346, Sec. 3.2 Eq. 4) defines

    A^(i,l) = (R^(i,l) - mean_all) / std_all + (R^(i,l) - mean_l) / std_l

where the first term normalises across *all* rollouts of the two equivalent states and the
second normalises inside one state.  Appendix D.1 works a full example out by hand; those
numbers are the golden values below (``PAPER_EXAMPLE`` in ``trainers.j4r.eis``).

The tests are written against the *definition*, not against the implementation, so that a
future refactor cannot quietly replace EIS with vanilla prompt-local normalisation: the
last class asserts that the two are numerically different on the same rewards.
"""

from __future__ import annotations

import math
import unittest

from trainers.j4r import eis


def _rollouts(rewards_by_state, *, pair_id="p1", start=0):
    """Build rollouts from ``{"state": [r0, r1, ...]}`` with per-state indices from 0."""
    out = []
    for state_id, rewards in rewards_by_state.items():
        for index, reward in enumerate(rewards):
            out.append(eis.Rollout(pair_id, state_id, index, float(reward)))
    return out


class PaperExampleTest(unittest.TestCase):
    """Appendix D.1: state A has 12x1.5 + 4x1.0, state B has 12x0.0 + 4x1.0."""

    def test_eis_values_match_the_hand_computed_example(self):
        computed = eis.compute_paper_example("eis")
        expected = eis.paper_example("eis")
        self.assertEqual(set(computed), set(expected))
        for key, value in expected.items():
            self.assertAlmostEqual(
                computed[key], value, delta=eis.PAPER_TOLERANCE,
                msg="EIS advantage for {0}".format(key))

    def test_global_only_matches_the_hand_computed_example(self):
        computed = eis.compute_paper_example("global_only")
        for key, value in eis.paper_example("global_only").items():
            self.assertAlmostEqual(computed[key], value, delta=eis.PAPER_TOLERANCE,
                                   msg="global term for {0}".format(key))

    def test_local_only_reproduces_plain_per_state_normalisation(self):
        computed = eis.compute_paper_example("separate")
        for key, value in eis.paper_example("separate").items():
            self.assertAlmostEqual(computed[key], value, delta=eis.PAPER_TOLERANCE,
                                   msg="local term for {0}".format(key))

    def test_the_paper_example_uses_unbiased_standard_deviations(self):
        # The published numbers only come out with the *population* standard deviation
        # (divide by N).  This pins that choice: a sample std would shift every value.
        rewards = [1.5] * 12 + [1.0] * 4
        mean = sum(rewards) / len(rewards)
        population = math.sqrt(sum((r - mean) ** 2 for r in rewards) / len(rewards))
        sample = math.sqrt(sum((r - mean) ** 2 for r in rewards) / (len(rewards) - 1))
        self.assertNotAlmostEqual(population, sample, places=3)
        self.assertAlmostEqual((1.0 - mean) / population, eis.paper_example("separate")[("A", 1.0)],
                               delta=eis.PAPER_TOLERANCE)

    def test_the_global_term_ignores_which_state_a_rollout_belongs_to(self):
        expected = eis.paper_example("global_only")
        # 1.0 appears in both states and must get the same global term in both.
        self.assertAlmostEqual(expected[("A", 1.0)], expected[("B", 1.0)],
                               delta=eis.PAPER_TOLERANCE)

    def test_eis_is_the_sum_of_the_two_terms(self):
        components = {("A", 1.5), ("A", 1.0), ("B", 1.0), ("B", 0.0)}
        eis_values = eis.compute_paper_example("eis")
        global_values = eis.compute_paper_example("global_only")
        local_values = eis.compute_paper_example("separate")
        for key in components:
            self.assertAlmostEqual(eis_values[key], global_values[key] + local_values[key],
                                   delta=eis.PAPER_TOLERANCE, msg=str(key))


class GroupingTest(unittest.TestCase):
    """Three-level identity: pair -> state -> rollout."""

    def test_group_rollouts_nests_pairs_states_and_rollouts(self):
        rollouts = _rollouts({"original": [1.0, 0.0], "swapped": [0.0, 1.0]})
        grouped = eis.group_rollouts(rollouts)
        self.assertEqual(set(grouped), {"p1"})
        self.assertEqual(set(grouped["p1"]), {"original", "swapped"})
        self.assertEqual(len(grouped["p1"]["original"]), 2)
        self.assertEqual(grouped["p1"]["original"][0].rollout_index, 0)

    def test_advantages_are_keyed_by_the_full_three_level_identity(self):
        rollouts = _rollouts({"original": [1.0, 0.0], "swapped": [0.0, 1.0]})
        result = eis.compute_eis_advantages(rollouts)
        keys = eis.advantages_by_key(result)
        self.assertEqual(set(keys), {("p1", "original", 0), ("p1", "original", 1),
                                     ("p1", "swapped", 0), ("p1", "swapped", 1)})

    def test_two_pairs_are_normalised_independently(self):
        rollouts = _rollouts({"original": [1.0, 1.0], "swapped": [1.0, 1.0]}, pair_id="p1")
        rollouts += _rollouts({"original": [1.0, 0.0], "swapped": [0.0, 1.0]}, pair_id="p2")
        result = eis.compute_eis_advantages(rollouts)
        keys = eis.advantages_by_key(result)
        self.assertEqual(result["num_pairs"], 2)
        # p1 is flat, so every advantage there is exactly zero.
        self.assertEqual(keys[("p1", "original", 0)], 0.0)
        self.assertEqual(keys[("p1", "swapped", 1)], 0.0)
        # p2 is not flat, so its advantages are non-zero.
        self.assertNotEqual(keys[("p2", "original", 0)], 0.0)

    def test_rollout_index_only_has_to_be_unique_within_a_state(self):
        # Both states number their rollouts from zero; that is what the sampler produces.
        rollouts = _rollouts({"original": [1.0, 0.0], "swapped": [0.0, 1.0]})
        result = eis.compute_eis_advantages(rollouts)
        self.assertEqual(result["num_rollouts"], 4)

    def test_a_repeated_three_level_key_is_rejected(self):
        rollouts = _rollouts({"original": [1.0, 0.0]})
        rollouts.append(eis.Rollout("p1", "original", 0, 0.5))
        with self.assertRaises(eis.EISError) as ctx:
            eis.compute_eis_advantages(rollouts)
        self.assertIn("duplicate", str(ctx.exception).lower())

    def test_an_empty_batch_is_rejected(self):
        with self.assertRaises(eis.EISError):
            eis.compute_eis_advantages([])


class DegenerateGroupTest(unittest.TestCase):
    """Zero variance is expected in judging (a pair can be all-correct), not an error."""

    def test_all_rewards_equal_yields_zero_advantages(self):
        rollouts = _rollouts({"original": [1.0, 1.0, 1.0], "swapped": [1.0, 1.0, 1.0]})
        result = eis.compute_eis_advantages(rollouts)
        keys = eis.advantages_by_key(result)
        self.assertTrue(all(value == 0.0 for value in keys.values()))
        self.assertEqual(result["zero_variance_global_pairs"], 1)

    def test_zero_variance_is_reported_per_state(self):
        rollouts = _rollouts({"original": [1.0, 1.0], "swapped": [0.0, 1.0]})
        result = eis.compute_eis_advantages(rollouts)
        per_state = result["pairs"]["p1"]["per_state"]
        self.assertTrue(per_state["original"]["zero_variance"])
        self.assertFalse(per_state["swapped"]["zero_variance"])
        self.assertEqual(result["zero_variance_state_count"], 1)

    def test_no_advantage_is_nan_or_infinite(self):
        for rewards in (
            {"original": [0.0, 0.0], "swapped": [0.0, 0.0]},
            {"original": [1.5, 1.5], "swapped": [1.5, 1.5]},
            {"original": [-0.5, 1.5], "swapped": [1.5, -0.5]},
        ):
            result = eis.compute_eis_advantages(_rollouts(rewards))
            for value in eis.advantages_by_key(result).values():
                self.assertTrue(math.isfinite(value), msg=str(rewards))

    def test_negative_rewards_are_handled(self):
        # J4R's -0.5 format penalty is a real reward value, so no clamp may exist.
        result = eis.compute_eis_advantages(
            _rollouts({"original": [-0.5, 1.5], "swapped": [1.5, -0.5]}))
        values = list(eis.advantages_by_key(result).values())
        self.assertTrue(all(math.isfinite(value) for value in values))
        self.assertGreater(max(values), 0.0)


class StateSwapInvarianceTest(unittest.TestCase):
    """The two states are equivalent, so swapping which one is 'original' changes nothing."""

    def test_swapping_state_labels_leaves_the_multiset_of_advantages_unchanged(self):
        forward = eis.advantages_by_key(eis.compute_eis_advantages(
            _rollouts({"original": [1.5, 1.0, 1.5], "swapped": [0.0, 1.0, 1.5]})))
        backward = eis.advantages_by_key(eis.compute_eis_advantages(
            _rollouts({"swapped": [1.5, 1.0, 1.5], "original": [0.0, 1.0, 1.5]})))
        self.assertEqual(sorted(forward.values()), sorted(backward.values()))

    def test_swapping_does_not_change_the_mean_or_the_std(self):
        forward = eis.compute_eis_advantages(
            _rollouts({"original": [1.5, 1.0], "swapped": [0.0, 1.0]}))
        backward = eis.compute_eis_advantages(
            _rollouts({"original": [0.0, 1.0], "swapped": [1.5, 1.0]}))
        self.assertAlmostEqual(forward["pairs"]["p1"]["global_mean"],
                               backward["pairs"]["p1"]["global_mean"])
        self.assertAlmostEqual(forward["pairs"]["p1"]["global_std"],
                               backward["pairs"]["p1"]["global_std"])

    def test_a_flat_pair_stays_flat_after_the_swap(self):
        result = eis.compute_eis_advantages(
            _rollouts({"original": [1.0, 1.0], "swapped": [1.0, 1.0]}))
        self.assertTrue(all(value == 0.0
                            for value in eis.advantages_by_key(result).values()))


class WrongGroupSizeTest(unittest.TestCase):
    """The expected shape is asserted, so a broken sampler cannot pass silently."""

    def test_too_few_states_is_rejected(self):
        rollouts = _rollouts({"original": [1.0, 0.0]})
        with self.assertRaises(eis.EISError) as ctx:
            eis.compute_eis_advantages(rollouts, expected_states_per_pair=2)
        self.assertIn("state", str(ctx.exception).lower())

    def test_too_many_states_is_rejected(self):
        rollouts = _rollouts({"original": [1.0, 0.0], "swapped": [0.0, 1.0],
                              "third": [0.0, 1.0]})
        with self.assertRaises(eis.EISError):
            eis.compute_eis_advantages(rollouts, expected_states_per_pair=2)

    def test_wrong_rollouts_per_state_is_rejected(self):
        rollouts = _rollouts({"original": [1.0, 0.0, 1.0], "swapped": [0.0, 1.0, 0.0]})
        with self.assertRaises(eis.EISError) as ctx:
            eis.compute_eis_advantages(rollouts, expected_rollouts_per_state=2)
        self.assertIn("rollout", str(ctx.exception).lower())

    def test_a_single_rollout_state_is_rejected(self):
        rollouts = _rollouts({"original": [1.0], "swapped": [0.0, 1.0]})
        with self.assertRaises(eis.EISError):
            eis.compute_eis_advantages(rollouts)


class NotVanillaNormalisationTest(unittest.TestCase):
    """EIS must differ from prompt-local normalisation on the same rewards.

    If someone replaces the EIS mixin with TRL's default ``scale_rewards="group"``, the
    per-state values below are what the model would be trained on.  The two must not agree
    on a non-degenerate example, otherwise the method has silently become vanilla GRPO.
    """

    def test_eis_differs_from_vanilla_prompt_local_normalisation(self):
        rewards = {"original": [1.0, 1.0, 0.0, 0.0], "swapped": [1.0, 1.0, 1.0, 1.0]}
        result = eis.compute_eis_advantages(_rollouts(rewards))
        eis_values = eis.advantages_by_key(result)

        def vanilla(values):
            mean = sum(values) / len(values)
            std = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))
            return [(v - mean) / (std + 1e-4) for v in values]

        vanilla_original = vanilla(rewards["original"])
        self.assertNotAlmostEqual(eis_values[("p1", "original", 0)], vanilla_original[0])

    def test_the_global_term_is_not_the_state_mean(self):
        # A rollout in a flat-but-low state gets a strongly negative global term even
        # though its own state mean equals its reward.
        result = eis.compute_eis_advantages(
            _rollouts({"original": [0.0, 0.0], "swapped": [1.5, 1.5]}))
        keys = eis.advantages_by_key(result)
        self.assertLess(keys[("p1", "original", 0)], 0.0)
        self.assertGreater(keys[("p1", "swapped", 0)], 0.0)

    def test_a_state_with_no_internal_spread_still_gets_a_global_signal(self):
        result = eis.compute_eis_advantages(
            _rollouts({"original": [0.0, 0.0], "swapped": [1.0, 1.0]}))
        keys = eis.advantages_by_key(result)
        # Local term is 0 (zero variance) but the global term is not.
        self.assertNotEqual(keys[("p1", "original", 0)], 0.0)


if __name__ == "__main__":
    unittest.main()
