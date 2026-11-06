"""Regression tests for the two defects that `--dry-run` exposed.

1. `build_sampler` crashed whenever `--limit` was smaller than `global_batch_size`.
   The fix narrows the batch and records the narrowing, so a debug plan can never
   masquerade as the controlled comparison budget.

2. `run_cli` built the trainer class before honouring `--dry-run`, so the J1/J4R
   entry points could not print a plan in an environment without TRL -- which is
   the only environment these CPU tests run in.
"""

from __future__ import annotations

import json
import os
import unittest

from common import config as config_mod
from data import paired_sampler
from trainers import grpo_base


METHODS = ("vanilla", "j1", "j4r", "jwc")
CONFIG_FILES = {
    "vanilla": "vanilla.yaml",
    "j1": "j1.yaml",
    "j4r": "j4r.yaml",
    "jwc": "jwc_rl_brier.yaml",
}


def _load(method):
    return config_mod.load_config(
        os.path.join(str(config_mod.paths.CONFIGS_DIR), CONFIG_FILES[method]))


class LimitedPlanTest(unittest.TestCase):
    """`--limit` is a debug switch, so it must work below one generation batch."""

    _full_plan = None

    @classmethod
    def full_plan(cls):
        """The un-narrowed plan, built once: it reads the whole 22k-row dataset."""
        if cls._full_plan is None:
            cls._full_plan = grpo_base.plan_run(_load("vanilla"),
                                                limit=_load("vanilla").global_batch_size)
        return cls._full_plan

    def test_a_short_limit_still_builds_a_sampler(self):
        cfg = _load("vanilla")
        rows, _ = grpo_base.load_rows(cfg, limit=4)
        sampler = grpo_base.build_sampler(cfg, rows)
        self.assertTrue(sampler.batch_narrowed)
        self.assertEqual(sampler.pairs_per_generation, 4)
        self.assertEqual(sampler.declared_pairs_per_generation, cfg.global_batch_size)

    def test_the_narrowed_sampler_does_not_claim_the_declared_batch(self):
        cfg = _load("vanilla")
        rows, _ = grpo_base.load_rows(cfg, limit=4)
        sampler = grpo_base.build_sampler(cfg, rows)
        # 4 pairs * 8 generations, not the configured 8 pairs * 8 generations.
        self.assertEqual(sampler.generation_batch_size, 4 * cfg.num_generations)
        self.assertNotEqual(sampler.generation_batch_size, cfg.rollouts_per_update)

    def test_the_plan_says_the_budget_is_not_comparable(self):
        cfg = _load("vanilla")
        plan = grpo_base.plan_run(cfg, limit=4)
        self.assertTrue(plan["debug"]["batch_narrowed"])
        self.assertFalse(plan["debug"]["budget_comparable"])
        self.assertEqual(plan["debug"]["limit"], 4)
        self.assertEqual(plan["debug"]["declared_pairs_per_generation"],
                         cfg.global_batch_size)
        self.assertEqual(plan["debug"]["effective_pairs_per_generation"], 4)
        self.assertIn("NOT the controlled comparison budget", plan["debug"]["note"])
        # The configured budget block is untouched: narrowing affects the sampler,
        # never the recorded comparison budget.
        self.assertEqual(plan["budget"]["total_rollouts"], cfg.total_rollouts_derived)

    def test_a_full_batch_is_not_marked_as_narrowed(self):
        cfg = _load("vanilla")
        plan = self.full_plan()
        self.assertFalse(plan["debug"]["batch_narrowed"])
        self.assertTrue(plan["debug"]["budget_comparable"])
        self.assertEqual(plan["debug"]["effective_pairs_per_generation"],
                         cfg.global_batch_size)
        self.assertEqual(plan["sampler"]["generation_batch_size"], cfg.rollouts_per_update)

    def test_every_method_can_plan_a_short_limit(self):
        for method in METHODS:
            with self.subTest(method=method):
                plan = grpo_base.plan_run(_load(method), limit=4)
                self.assertEqual(plan["method"], method)
                self.assertTrue(plan["debug"]["batch_narrowed"])

    def test_a_limit_of_exactly_one_batch_is_not_narrowed(self):
        # Boundary case: a limit equal to global_batch_size is a full batch, so
        # nothing is narrowed and the budget stays comparable.
        cfg = _load("vanilla")
        plan = grpo_base.plan_run(cfg, limit=cfg.global_batch_size)
        self.assertFalse(plan["debug"]["batch_narrowed"])
        self.assertTrue(plan["debug"]["budget_comparable"])
        self.assertEqual(plan["debug"]["effective_pairs_per_generation"],
                         cfg.global_batch_size)

    def test_narrow_to_layout_refuses_to_widen(self):
        cfg = _load("vanilla")
        rows, _ = grpo_base.load_rows(cfg, limit=4)
        sampler = grpo_base.build_sampler(cfg, rows)
        with self.assertRaises(paired_sampler.SamplerError) as ctx:
            sampler.narrow_to_layout(2)
        self.assertIn("cannot widen", str(ctx.exception))


class DryRunWithoutTrlTest(unittest.TestCase):
    """`--dry-run` must print a plan with no training stack installed.

    The environment these tests run in has no `trl`, so a passing test here is the
    assertion: if `run_cli` still called the trainer-class factory on a dry run,
    this file could not even import the J1/J4R entry points to completion.
    """

    def _run_cli(self, method, extra=("--limit", "4")):
        module = __import__("trainers.{0}.trl_trainer".format(method),
                            fromlist=["main"])
        argv = ["--config",
                os.path.join(str(config_mod.paths.CONFIGS_DIR), CONFIG_FILES[method]),
                "--dry-run"] + list(extra)
        # `--limit 4` keeps the four entry-point tests off the 22k-row file; the
        # point of these cases is that no TRL is imported, not the batch size.
        # `main` prints the plan; capture it by swapping stdout for a buffer.
        import io
        import sys
        buffer = io.StringIO()
        original = sys.stdout
        sys.stdout = buffer
        try:
            code = module.main(argv)
        finally:
            sys.stdout = original
        return code, json.loads(buffer.getvalue())

    def test_the_j1_entry_point_plans_without_trl(self):
        code, plan = self._run_cli("j1")
        self.assertEqual(code, 0)
        self.assertEqual(plan["method"], "j1")
        self.assertEqual(plan["budget"]["states_per_sample"], 2)

    def test_the_j4r_entry_point_plans_without_trl(self):
        code, plan = self._run_cli("j4r")
        self.assertEqual(code, 0)
        self.assertTrue(plan["advantage"]["evaluated_inside_trainer"])
        self.assertIn("EIS", plan["advantage"]["description"])

    def test_the_j4r_plan_records_eis_without_building_the_trainer(self):
        # The plan carries the advantage identity, so skipping the trainer class
        # does not lose the one fact that makes J4R distinct.
        _, plan = self._run_cli("j4r")
        self.assertIn("mean_all", plan["advantage"]["description"])
        self.assertIn("mean_state", plan["advantage"]["description"])

    def test_all_four_entry_points_plan_without_trl(self):
        for method in METHODS:
            with self.subTest(method=method):
                code, plan = self._run_cli(method)
                self.assertEqual(code, 0)
                self.assertEqual(plan["method"], method)
                self.assertEqual(plan["output_format"], method)

    def test_a_dry_run_prints_json_only(self):
        _, plan = self._run_cli("vanilla")
        self.assertIsInstance(plan, dict)
        json.dumps(plan)  # must be serialisable, the shell scripts parse it


if __name__ == "__main__":
    unittest.main()
