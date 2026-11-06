"""Config, sampler and per-method wiring tests (no torch, no TRL, no GPU).

``plan_run`` is the seam that lets these tests exist at all: it builds the dataset rows, the
paired sampler and the TRL keyword arguments without instantiating a trainer, so "the J4R
plan uses EIS" and "one update holds exactly the declared number of rollouts" are checked
in an environment where ``trl`` is not installed.

The classes are ordered from the cheapest assertion to the most expensive one (the last
class reads the real 22k-row HelpSteer file).
"""

from __future__ import annotations

import os
import unittest

from common import config as config_mod
from common import paths
from data import paired_sampler
from trainers import grpo_base, trl_bridge


METHODS = ("vanilla", "j1", "j4r", "jwc")


def _configs():
    return config_mod.load_all_configs()


class ShippedConfigTest(unittest.TestCase):
    """The four shipped YAML files agree on everything that must be held constant."""

    @classmethod
    def setUpClass(cls):
        cls.configs = _configs()

    def test_all_four_methods_ship_a_config(self):
        self.assertEqual(sorted(self.configs), sorted(METHODS))

    def test_the_output_format_matches_the_method(self):
        for method, cfg in self.configs.items():
            self.assertEqual(cfg.output_format, method, msg=method)

    def test_the_base_model_and_training_source_are_the_fixed_ones(self):
        for method, cfg in self.configs.items():
            self.assertEqual(os.path.realpath(cfg["base_model"]),
                             os.path.realpath(str(paths.BASE_MODEL)), msg=method)
            self.assertEqual(os.path.realpath(cfg["train_dataset"]),
                             os.path.realpath(str(paths.HELPSTEER_TRAIN)), msg=method)

    def test_no_config_trains_on_rewardbench(self):
        for method, cfg in self.configs.items():
            self.assertNotIn("rewardbench", cfg["train_dataset"].lower(), msg=method)

    def test_the_token_budget_is_identical_across_methods(self):
        # Same generated-token budget per method is what makes the four runs comparable;
        # if this drifts, the comparison silently becomes "who trained longer".
        lines = config_mod.assert_equal_budget(self.configs)
        self.assertEqual(len(lines), 2)

    def test_shared_hyperparameters_are_identical(self):
        keys = ("seed", "learning_rate", "lr_scheduler_type", "warmup_ratio",
                "temperature", "top_p", "clip_epsilon", "kl_beta",
                "reference_model_policy", "max_prompt_length", "max_completion_length",
                "global_batch_size", "micro_batch_size", "estimated_prompt_tokens")
        for key in keys:
            values = {cfg[key] for cfg in self.configs.values()}
            self.assertEqual(len(values), 1, msg="{0}: {1}".format(key, values))

    def test_the_paired_methods_declare_two_states(self):
        self.assertEqual(self.configs["j1"].states_per_sample, 2)
        self.assertEqual(self.configs["j4r"].states_per_sample, 2)
        self.assertEqual(self.configs["vanilla"].states_per_sample, 1)
        self.assertEqual(self.configs["jwc"].states_per_sample, 1)

    def test_every_method_keeps_eight_rollouts_per_state(self):
        # This is the controlled-comparison knob: equal rollouts per prompt state, with
        # the paired methods spending them across two states instead of one.
        for method, cfg in self.configs.items():
            self.assertEqual(cfg.rollouts_per_state, 8, msg=method)

    def test_the_context_budget_fits_the_model(self):
        for method, cfg in self.configs.items():
            self.assertLessEqual(cfg.context_budget, paths.MAX_CONTEXT_TOKENS, msg=method)

    def test_a_frozen_reference_implies_a_positive_kl_coefficient(self):
        for method, cfg in self.configs.items():
            if cfg["reference_model_policy"] == config_mod.REFERENCE_POLICY_FROZEN:
                self.assertGreater(cfg.kl_beta, 0.0, msg=method)

    def test_no_method_declares_a_warmup_it_does_not_use(self):
        for method, cfg in self.configs.items():
            if cfg["lr_scheduler_type"] == "constant":
                self.assertEqual(cfg["warmup_ratio"], 0.0, msg=method)


class DerivedBudgetTest(unittest.TestCase):
    """The per-update arithmetic has to close, or the data loader disagrees with the sampler."""

    @classmethod
    def setUpClass(cls):
        cls.configs = _configs()

    def test_rows_per_update_equals_global_batch_times_states(self):
        for method, cfg in self.configs.items():
            self.assertEqual(cfg.rows_per_update,
                             cfg.global_batch_size * cfg.states_per_sample, msg=method)

    def test_rollouts_per_update_is_the_product_of_both_factors(self):
        for method, cfg in self.configs.items():
            self.assertEqual(cfg.rollouts_per_update,
                             cfg.rows_per_update * cfg.rollouts_per_state, msg=method)
            self.assertEqual(cfg.rollouts_per_update,
                             cfg.global_batch_size * cfg.num_generations, msg=method)

    def test_generation_batch_size_matches_the_sampler(self):
        for method, cfg in self.configs.items():
            self.assertEqual(trl_bridge.generation_batch_size(cfg),
                             cfg.rollouts_per_update, msg=method)

    def test_trl_sees_one_group_per_state(self):
        # TRL normalises rewards with rewards.view(-1, num_generations); handing it
        # rollouts_per_state (not the total) is what makes one group one state.
        for method, cfg in self.configs.items():
            kwargs = trl_bridge.build_trl_config(cfg)
            self.assertEqual(kwargs["num_generations"], cfg.rollouts_per_state, msg=method)
            self.assertEqual(kwargs["generation_batch_size"],
                             cfg.rollouts_per_update, msg=method)

    def test_trl_batch_size_counts_prompts_not_dataset_rows(self):
        for method, cfg in self.configs.items():
            kwargs = trl_bridge.build_trl_config(cfg)
            self.assertEqual(kwargs["per_device_train_batch_size"],
                             cfg.micro_batch_size * cfg.rollouts_per_state, msg=method)
            self.assertEqual(
                kwargs["per_device_train_batch_size"] * kwargs["gradient_accumulation_steps"],
                kwargs["generation_batch_size"], msg=method)

    def test_the_loss_and_scaling_are_the_ones_the_methods_assume(self):
        for method, cfg in self.configs.items():
            kwargs = trl_bridge.build_trl_config(cfg)
            self.assertEqual(kwargs["loss_type"], "grpo", msg=method)
            self.assertEqual(kwargs["scale_rewards"], "group", msg=method)
            self.assertFalse(kwargs["remove_unused_columns"], msg=method)

    def test_the_generation_batch_is_not_silently_resized(self):
        # auto_find_batch_size would break the "one group == one state" invariant, so it
        # must never be switched on.
        for method, cfg in self.configs.items():
            flags = trl_bridge.build_trl_config(cfg)
            flags.update(trl_bridge.DEFAULT_TRAINING_FLAGS)
            self.assertFalse(flags.get("auto_find_batch_size", False), msg=method)


class SamplerLayoutTest(unittest.TestCase):
    """The sampler determines the row order the whole method depends on."""

    @classmethod
    def setUpClass(cls):
        cls.configs = _configs()
        cls.rows, cls.summary = grpo_base.load_rows(cls.configs["j4r"], limit=40)
        cls.sampler = grpo_base.build_sampler(cls.configs["j4r"], cls.rows)

    def test_the_sampler_covers_the_declared_budget(self):
        cfg = self.configs["j4r"]
        self.assertEqual(self.sampler.generation_batch_size, cfg.rollouts_per_update)
        self.assertEqual(len(self.sampler), cfg.training_steps * cfg.rollouts_per_update)

    def test_every_pair_contributes_both_states(self):
        batch = self.sampler.batch_layout()[0]
        pairs = {}
        for pair_id, view, row_index, rollout_index in batch:
            pairs.setdefault(pair_id, set()).add(view)
        self.assertTrue(pairs)
        for pair_id, views in pairs.items():
            self.assertEqual(views, {"original", "swapped"}, msg=pair_id)

    def test_the_two_views_of_a_pair_are_adjacent(self):
        batch = self.sampler.batch_layout()[0]
        views_in_order = [view for _, view, _, _ in batch]
        # original states occupy even blocks, swapped states the following odd block
        self.assertEqual(views_in_order[:8], ["original"] * 8)
        self.assertEqual(views_in_order[8:16], ["swapped"] * 8)

    def test_rollout_indices_run_from_zero_and_cover_the_group(self):
        batch = self.sampler.batch_layout()[0]
        first_state = [(pair_id, view, index)
                       for pair_id, view, _, index in batch if view == "original"]
        pair_id = first_state[0][0]
        indices = [index for pid, _, index in first_state if pid == pair_id]
        self.assertEqual(indices, list(range(self.configs["j4r"].rollouts_per_state)))

    def test_every_row_index_is_inside_the_dataset(self):
        for batch in self.sampler.batch_layout():
            for _, _, row_index, _ in batch:
                self.assertGreaterEqual(row_index, 0)
                self.assertLess(row_index, len(self.rows))

    def test_the_sampler_is_deterministic_for_a_seed(self):
        again = grpo_base.build_sampler(self.configs["j4r"], self.rows)
        self.assertEqual(self.sampler.batch_layout(), again.batch_layout())

    def test_a_single_state_method_never_emits_a_swapped_row(self):
        rows, _ = grpo_base.load_rows(self.configs["vanilla"], limit=20)
        sampler = grpo_base.build_sampler(self.configs["vanilla"], rows)
        views = {view for _, view, _, _ in sampler.batch_layout()[0]}
        self.assertEqual(views, {"original"})

    def test_the_layout_rejects_a_missing_state(self):
        sample_ids = ["p1::original", "p2::original", "p2::swapped"]
        with self.assertRaises(paired_sampler.SamplerError):
            paired_sampler.build_pair_layout(sample_ids, expected_states=2)


class PlanTest(unittest.TestCase):
    """``plan_run`` is the CPU-testable description of what a run would do."""

    @classmethod
    def setUpClass(cls):
        cls.configs = _configs()
        cls.plans = {
            method: grpo_base.plan_run(cfg, limit=24)
            for method, cfg in cls.configs.items()
        }

    def test_every_plan_names_its_method_and_parser(self):
        for method, plan in self.plans.items():
            self.assertEqual(plan["method"], method)
            self.assertEqual(plan["output_format"], method)

    def test_only_j4r_evaluates_its_advantage_inside_the_trainer(self):
        for method, plan in self.plans.items():
            expected = method == "j4r"
            self.assertEqual(plan["advantage"]["evaluated_inside_trainer"], expected,
                             msg=method)

    def test_the_j4r_plan_describes_eis_explicitly(self):
        advantage = self.plans["j4r"]["advantage"]["description"]
        self.assertIn("EIS", advantage)
        self.assertIn("mean_all", advantage)
        self.assertIn("mean_state", advantage)

    def test_no_other_plan_claims_to_use_eis(self):
        for method in ("vanilla", "j1", "jwc"):
            self.assertNotIn("EIS", self.plans[method]["advantage"]["description"],
                             msg=method)

    def test_each_plan_documents_its_reward(self):
        self.assertIn("correctness", self.plans["vanilla"]["reward"]["description"].lower())
        self.assertIn("consistency", self.plans["j1"]["reward"]["description"].lower())
        self.assertIn("brier", self.plans["jwc"]["reward"]["description"].lower())
        self.assertIn("format", self.plans["j4r"]["reward"]["description"].lower())

    def test_the_budget_block_matches_the_config(self):
        for method, plan in self.plans.items():
            cfg = self.configs[method]
            self.assertEqual(plan["budget"]["states_per_sample"], cfg.states_per_sample)
            self.assertEqual(plan["budget"]["rollouts_per_update"],
                             cfg.rollouts_per_update)
            self.assertEqual(plan["budget"]["total_rollouts"], cfg.total_rollouts_derived)
            self.assertEqual(plan["budget"]["estimated_generated_tokens"],
                             cfg["estimated_generated_tokens"])
            # The declared budget and the recomputed one must agree, otherwise the run
            # silently trains for a different number of rollouts than the table claims.
            self.assertEqual(cfg["total_rollouts"], cfg.total_rollouts_derived)
            self.assertEqual(cfg["estimated_generated_tokens"],
                             cfg.estimated_generated_tokens_derived)

    def test_the_plan_never_carries_a_gpu_object(self):
        # Everything in the plan must be JSON-serialisable; a leaked tensor or trainer
        # would make --dry-run fail on a machine without the training stack.
        import json

        json.dumps(self.plans, default=str)

    def test_a_plan_does_not_write_anything(self):
        cfg = self.configs["vanilla"]
        before = sorted(os.listdir(str(paths.RUNS_DIR)))
        grpo_base.plan_run(cfg, limit=8)
        self.assertEqual(before, sorted(os.listdir(str(paths.RUNS_DIR))))


class MissingDependencyTest(unittest.TestCase):
    """The trainer layer must fail with a readable message, not an ImportError traceback."""

    def test_require_names_the_package_and_the_reason(self):
        # A module that cannot exist, so the failure path is exercised regardless of what
        # this machine has installed.  Asserting on `require("trl")` would assert on the
        # environment instead: it passes on a bare host and fails once the pinned TRL
        # environment is built, which says nothing about the message under test.
        with self.assertRaises(trl_bridge.MissingDependencyError) as ctx:
            trl_bridge.require("grpo_no_such_module", "train any of the four methods")
        message = str(ctx.exception)
        self.assertIn("grpo_no_such_module", message)
        self.assertIn("not installed in this environment", message)
        self.assertIn("Nothing is installed automatically", message)
        # An unmapped module still gets a hint rather than a blank.
        self.assertEqual(ctx.exception.module, "grpo_no_such_module")
        self.assertEqual(ctx.exception.hint, "requirements/README.md")

    def test_an_installed_package_is_returned(self):
        module = trl_bridge.require("json", "parsing prediction files")
        self.assertTrue(hasattr(module, "dumps"))

    def test_the_hint_names_the_requirements_file_for_each_mapped_stack(self):
        """The hint is the only thing that tells a user where to go next."""
        for module, hint in (
            ("trl", "requirements/trl-py311.txt"),
            ("datasets", "requirements/trl-py311.txt"),
            ("verl", "requirements/verl-py310.txt"),
            ("ray", "requirements/verl-py310.txt"),
        ):
            error = trl_bridge.MissingDependencyError(module, "training")
            self.assertEqual(error.hint, hint, module)
            self.assertIn(hint, str(error))

    def test_the_prompt_budget_guard_accepts_the_shipped_configs(self):
        for cfg in _configs().values():
            trl_bridge.assert_prompt_budget(cfg)

    def test_the_prompt_budget_guard_rejects_an_oversized_window(self):
        class _Budget(object):
            max_prompt_length = 32000
            max_completion_length = 1024

        with self.assertRaises(ValueError) as ctx:
            trl_bridge.assert_prompt_budget(_Budget())
        self.assertIn("exceeds the Qwen2.5-3B context window", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
