"""Composition order: the rollout hook must run *after* the method's own hook.

``build_trainer`` builds ``class _Trainer(SamplerMixin, RolloutLogMixin, base)`` where
``base`` is either ``GRPOTrainer`` or a method subclass (``J4RTrainer`` for J4R).  That
puts ``RolloutLogMixin`` *before* the method class in the MRO, so its
``_generate_and_score_completions`` wraps the method's -- which is what we want, but it
is exactly the kind of ordering that breaks silently: if it were reversed, J4R would
record rollout health for advantages that had not been replaced by EIS yet, and the
numbers would describe the wrong thing.

These tests pin the ordering with stand-in bases, so they run without torch/trl.
"""
from __future__ import annotations

import unittest

from trainers import grpo_base


class _RecordingBase(object):
    """Stand-in for GRPOTrainer: records that it generated."""

    def __init__(self):
        self.calls = []
        self.reward_funcs = []
        self.state = None

    def _generate_and_score_completions(self, inputs):
        self.calls.append("generate")
        return {"advantages": "vanilla-normalised", "inputs": inputs}


class _MethodMixin(object):
    """Stand-in for EISAdvantageMixin: replaces the advantages."""

    def _generate_and_score_completions(self, inputs):
        output = super()._generate_and_score_completions(inputs)
        self.calls.append("method")
        output["advantages"] = "method-normalised"
        return output


class _MethodTrainer(_MethodMixin, _RecordingBase):
    """Stand-in for J4RTrainer."""


class _Composed(grpo_base.SamplerMixin, grpo_base.RolloutLogMixin, _MethodTrainer):
    """Exactly the class ``build_trainer`` creates, with fake bases."""


class _ComposedPlain(grpo_base.SamplerMixin, grpo_base.RolloutLogMixin, _RecordingBase):
    """The vanilla/J1/JwC shape."""


class CompositionOrderTest(unittest.TestCase):
    def test_the_rollout_hook_wraps_the_method_hook(self):
        trainer = _Composed()
        output = trainer._generate_and_score_completions({"row": 0})
        # generate -> method (EIS) -> then the rollout logger observes the finished batch.
        self.assertEqual(trainer.calls, ["generate", "method"])
        self.assertEqual(output["advantages"], "method-normalised")

    def test_the_logger_sees_the_advantages_the_method_produced(self):
        """If the hook ran first, it would see the pre-EIS advantages."""
        seen = {}

        class _Spy(_Composed):
            def log_rollout_health(self):
                # By the time we are called, the method mixin has already run.
                seen["calls"] = list(self.calls)
                return None

        _Spy()._generate_and_score_completions({})
        self.assertEqual(seen["calls"], ["generate", "method"])

    def test_a_plain_method_still_composes(self):
        trainer = _ComposedPlain()
        output = trainer._generate_and_score_completions({"row": 1})
        self.assertEqual(trainer.calls, ["generate"])
        self.assertEqual(output["advantages"], "vanilla-normalised")

    def test_a_failing_log_does_not_abort_training(self):
        class _Broken(_ComposedPlain):
            def log_rollout_health(self):
                raise RuntimeError("diagnostics exploded")

        output = _Broken()._generate_and_score_completions({})
        self.assertEqual(output["advantages"], "vanilla-normalised")

    def test_the_hook_is_in_the_mro_before_the_method_class(self):
        mro = _Composed.__mro__
        self.assertLess(mro.index(grpo_base.RolloutLogMixin),
                        mro.index(_MethodMixin))


if __name__ == "__main__":
    unittest.main()
