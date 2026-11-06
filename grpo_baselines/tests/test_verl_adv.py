"""Tests for the verl advantage-estimator integration of J4R's EIS.

``trainers/j4r/verl_adv.py`` is the path a verl run would take instead of the TRL
subclass, and until now nothing imported it. The tests below run the real torch
tensor plumbing and the real ``register_adv_est`` protocol against a stand-in
``core_algos`` module, so verl does not have to be installed.

Two things about reading the output that are easy to get wrong, and that these
tests therefore pin down explicitly:

* the estimator returns ``(bs, response_length)``, with the sequence advantage
  broadcast over every unmasked token and zeroed on masked ones. Reading a row's
  advantage means taking **one** token column, not summing the row.
* a state with zero-variance rewards gets a *local* term of exactly zero, but its
  *global* term is whatever the pair-wide normalisation says. So a flat state is not
  a zero advantage -- that is the whole point of EIS, and
  ``test_a_zero_variance_state_still_carries_the_global_term`` asserts it.
"""

from __future__ import annotations

import inspect
import types
import unittest

from trainers.j4r.advantage import EISAdvantageError, compute_advantages
from trainers.j4r.eis import EPSILON, EISError
from trainers.j4r.verl_adv import (
    ADV_ESTIMATOR_NAME,
    UID_CONVENTION,
    _epsilon_from,
    einsum_advantage,
    estimator_config,
    register,
)

try:  # torch is present in the read-only environment; skip cleanly if it is not.
    import torch
except ImportError:  # pragma: no cover - the environment has it
    torch = None


# One pair, two states, three rollouts per state:
#   original : 1.0, 0.0, 1.0
#   swapped  : 1.0, 1.0, 0.0
# All six rewards: mean = 4/6 = 2/3, population sigma = sqrt(2/9) = 0.4714045,
# so the global term is +0.7071068 for a 1.0 and -1.4142136 for a 0.0.
# Within either state the multiset is {1, 1, 0}: mean = 2/3, same sigma,
# so the local term is also +0.7071068 for a 1.0 and -1.4142136 for a 0.0.
# EIS = global + local  =>  1.0 -> 1.4142136, 0.0 -> -2.8284271.
PAIR = "hs3-11"
UID = [PAIR + "::original"] * 3 + [PAIR + "::swapped"] * 3
SCORES = [1.0, 0.0, 1.0, 1.0, 1.0, 0.0]
EXPECTED = [1.4142135624, -2.8284271247, 1.4142135624,
            1.4142135624, 1.4142135624, -2.8284271247]


@unittest.skipIf(torch is None, "torch is not installed; the verl path needs real tensors")
class EinsumAdvantageTest(unittest.TestCase):
    """The estimator's numbers, shape and masking."""

    def _inputs(self, response_length=4, scores=None):
        scores = SCORES if scores is None else scores
        rewards = torch.zeros((len(scores), response_length), dtype=torch.float32)
        # Put the whole sequence return on the last token, the way verl's own
        # outcome-level estimators read it (scores = rewards.sum(dim=-1)).
        rewards[:, -1] = torch.tensor(scores, dtype=torch.float32)
        return rewards, torch.ones_like(rewards)

    def test_the_hand_worked_batch_is_reproduced_elementwise(self):
        rewards, mask = self._inputs()
        advantages, _ = einsum_advantage(rewards, mask, index=UID)
        # One token column is the sequence advantage; summing the row would count
        # the broadcast response_length times.
        values = advantages[:, -1].tolist()
        for index, (got, want) in enumerate(zip(values, EXPECTED)):
            self.assertAlmostEqual(
                got, want, delta=1e-6,
                msg="row {0} ({1}) should be {2}, got {3}".format(index, UID[index], want, got),
            )

    def test_the_hand_worked_values_match_the_trl_path(self):
        """Both integrations call compute_eis_advantages; the docstring says so.

        The comparison is float32 against float64: ``einsum_advantage`` builds its
        tensor with the *input* dtype, and the batch here is float32. The tolerance
        is therefore float32's, not 1e-9 -- a tighter bound would test the dtype
        rather than the arithmetic.
        """
        rewards, mask = self._inputs()
        advantages, _ = einsum_advantage(rewards, mask, index=UID)
        via_trl = compute_advantages(SCORES, UID)["advantages"]
        for index, got in enumerate(advantages[:, -1].tolist()):
            self.assertAlmostEqual(got, via_trl[index], delta=1e-6)
        self.assertEqual(advantages.dtype, rewards.dtype)

    def test_returns_is_a_copy_carrying_the_same_numbers(self):
        rewards, mask = self._inputs()
        advantages, returns = einsum_advantage(rewards, mask, index=UID)
        self.assertEqual(tuple(returns.shape), tuple(advantages.shape))
        self.assertTrue(torch.equal(returns, advantages))
        self.assertIsNot(returns, advantages)

    def test_advantages_are_broadcast_over_tokens_and_masked(self):
        rewards, _ = self._inputs(response_length=4)
        mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]] * 6)
        advantages, _ = einsum_advantage(rewards, mask, index=UID)
        self.assertEqual(tuple(advantages.shape), (6, 4))
        # Every unmasked token of a row carries that row's sequence advantage...
        self.assertAlmostEqual(advantages[0, 0].item(), EXPECTED[0], delta=1e-6)
        self.assertAlmostEqual(advantages[0, 1].item(), EXPECTED[0], delta=1e-6)
        # ...and masked positions are zero, not "whatever was there".
        self.assertEqual(advantages[0, 2].item(), 0.0)
        self.assertEqual(advantages[0, 3].item(), 0.0)

    def test_a_fully_flat_batch_is_all_zero_and_never_nan(self):
        """Every reward equal kills both normalisers; EIS must return 0.0, not NaN."""
        scores = [1.0] * 6
        rewards, mask = self._inputs(scores=scores)
        advantages, _ = einsum_advantage(rewards, mask, index=UID)
        values = advantages[:, -1].tolist()
        self.assertFalse(any(value != value for value in values), "advantages contained NaN")
        for value in values:
            self.assertEqual(value, 0.0)

    def test_a_zero_variance_state_still_carries_the_global_term(self):
        """original is flat at 0.0: local term 0, global term -0.7071068.

        This is the distinction a vanilla implementation would lose. A flat group
        is not a zero advantage under EIS; only a flat *pair* is.
        """
        scores = [0.0, 0.0, 0.0, 1.0, 0.0, 1.0]
        rewards, mask = self._inputs(scores=scores)
        advantages, _ = einsum_advantage(rewards, mask, index=UID)
        values = advantages[:, -1].tolist()
        self.assertFalse(any(value != value for value in values), "advantages contained NaN")
        for index in (0, 1, 2):
            self.assertAlmostEqual(values[index], -0.7071067812, delta=1e-6)
        result = compute_advantages(scores, UID)
        self.assertEqual(result["zero_variance_state_count"], 1)
        original = result["pairs"][PAIR]["per_state"]["original"]
        self.assertTrue(original["zero_variance"])
        self.assertEqual(original["std"], 0.0)


@unittest.skipIf(torch is None, "torch is not installed")
class IndexContractTest(unittest.TestCase):
    """verl hands over exactly one grouping column; that contract is enforced."""

    def _rewards(self, rows):
        rewards = torch.zeros((rows, 2), dtype=torch.float32)
        rewards[:, -1] = torch.arange(rows, dtype=torch.float32)
        return rewards, torch.ones_like(rewards)

    def test_missing_index_is_rejected_with_the_convention(self):
        rewards, mask = self._rewards(4)
        with self.assertRaises(ValueError) as caught:
            einsum_advantage(rewards, mask, index=None)
        message = str(caught.exception)
        self.assertIn("uid", message)
        self.assertIn(UID_CONVENTION, message)

    def test_index_of_the_wrong_length_is_rejected(self):
        rewards, mask = self._rewards(4)
        with self.assertRaises(ValueError) as caught:
            einsum_advantage(rewards, mask, index=["a::original"])
        self.assertIn("index has 1 entries", str(caught.exception))

    def test_uids_without_the_pair_state_shape_are_rejected(self):
        """Plain '0'/'1' uids must fail loudly, not degrade to per-group GRPO."""
        rewards, mask = self._rewards(4)
        with self.assertRaises(EISAdvantageError) as caught:
            einsum_advantage(rewards, mask, index=["0", "1", "2", "3"])
        self.assertIn("::", str(caught.exception))

    def test_a_misshapen_reward_tensor_is_rejected(self):
        rewards, mask = self._rewards(4)
        with self.assertRaises(ValueError) as caught:
            einsum_advantage(rewards.unsqueeze(0), mask, index=UID[:4])
        self.assertIn("(bs, response_length)", str(caught.exception))

    def test_a_mask_of_the_wrong_shape_is_rejected(self):
        rewards, mask = self._rewards(4)
        with self.assertRaises(ValueError) as caught:
            einsum_advantage(rewards, mask[:, :1], index=UID[:4])
        self.assertIn("does not match", str(caught.exception))

    def test_a_state_with_one_rollout_is_rejected(self):
        """verl must not silently accept a group the local term cannot normalise."""
        rewards = torch.zeros((2, 2), dtype=torch.float32)
        rewards[:, -1] = torch.tensor([1.0, 1.0])
        mask = torch.ones_like(rewards)
        with self.assertRaises(EISError) as caught:
            einsum_advantage(
                rewards, mask, index=[PAIR + "::original", PAIR + "::swapped"]
            )
        self.assertIn(">= 2", str(caught.exception))


@unittest.skipIf(torch is None, "torch is not installed")
class EpsilonOverrideTest(unittest.TestCase):
    """``_epsilon_from`` accepts a dict or an object, and ignores nonsense."""

    def test_a_flat_batch_is_zero_regardless_of_epsilon(self):
        rewards = torch.zeros((4, 2), dtype=torch.float32)
        rewards[:, -1] = torch.tensor([1.0, 1.0, 1.0, 1.0])
        mask = torch.ones_like(rewards)
        index = [PAIR + "::original"] * 2 + [PAIR + "::swapped"] * 2
        for config in (None, {}, {"epsilon": 1e-3}, types.SimpleNamespace(eis_epsilon=1e-3)):
            advantages, _ = einsum_advantage(rewards, mask, index=index, config=config)
            self.assertEqual(advantages[:, -1].tolist(), [0.0, 0.0, 0.0, 0.0])

    def test_a_positive_override_is_used_and_a_bad_one_is_ignored(self):
        self.assertEqual(_epsilon_from(None), EPSILON)
        self.assertEqual(_epsilon_from({}), EPSILON)
        self.assertEqual(_epsilon_from({"epsilon": 1e-3}), 1e-3)
        self.assertEqual(_epsilon_from(types.SimpleNamespace(adv_estimator_epsilon=0.5)), 0.5)
        self.assertEqual(_epsilon_from({"eis_epsilon": 0.25}), 0.25)
        self.assertEqual(_epsilon_from({"epsilon": 0.0}), EPSILON)         # not > 0
        self.assertEqual(_epsilon_from({"epsilon": -1.0}), EPSILON)        # not > 0
        self.assertEqual(_epsilon_from({"epsilon": "nonsense"}), EPSILON)  # not a float


class RegisterTest(unittest.TestCase):
    """The registry protocol, against a stand-in for verl's ``core_algos``."""

    def _fake_core_algos(self):
        registry = {}

        def register_adv_est(name):
            def decorator(function):
                registry[name] = function
                return function
            return decorator

        def get_adv_estimator_fn(name):
            return registry[name]

        module = types.ModuleType("fake_core_algos")
        module.ADV_ESTIMATOR_REGISTRY = registry
        module.register_adv_est = register_adv_est
        module.get_adv_estimator_fn = get_adv_estimator_fn
        return module

    def test_register_returns_the_function_under_the_declared_name(self):
        core_algos = self._fake_core_algos()
        returned = register(core_algos)
        self.assertIs(returned, einsum_advantage)
        self.assertIn(ADV_ESTIMATOR_NAME, core_algos.ADV_ESTIMATOR_REGISTRY)
        self.assertIs(
            core_algos.get_adv_estimator_fn(ADV_ESTIMATOR_NAME), einsum_advantage
        )

    def test_the_signature_is_what_verl_dispatch_requires(self):
        """verl inspects the signature to decide whether to pass ``index``."""
        parameters = inspect.signature(einsum_advantage).parameters
        for name in ("token_level_rewards", "response_mask", "config"):
            self.assertIn(name, parameters)
        self.assertIn("index", parameters)
        self.assertIsNone(parameters["index"].default)

    def test_register_without_verl_fails_loudly(self):
        """No silent no-op: calling register() in an environment without verl raises."""
        try:
            import verl.trainer.ppo.core_algos  # noqa: F401
        except ImportError:
            with self.assertRaises(ImportError):
                register()
        else:  # pragma: no cover - verl is not installed here
            self.assertIs(register(), einsum_advantage)


class EstimatorConfigTest(unittest.TestCase):
    def test_it_names_the_estimator_and_the_uid_convention(self):
        config = estimator_config()
        self.assertEqual(config["adv_estimator"], ADV_ESTIMATOR_NAME)
        self.assertEqual(config["uid_convention"], UID_CONVENTION)

    def test_the_default_name_is_j4r_eis(self):
        self.assertEqual(ADV_ESTIMATOR_NAME, "j4r_eis")
