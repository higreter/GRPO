"""Tests that the rollout diagnostics actually reach the trainer's log and the manifest."""
from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from common import config as config_mod
from common import paths
from rewards import correctness
from rewards import rollout_log as rl
from trainers import grpo_base


class _State(object):
    def __init__(self):
        self.log_history = []


class _FakeTrainer(grpo_base.RolloutLogMixin):
    """Stands in for GRPOTrainer: only what the mixin touches."""

    def __init__(self, reward_funcs):
        self.reward_funcs = reward_funcs
        self.state = _State()


class RolloutHealthMixinTest(unittest.TestCase):
    def test_a_flat_batch_is_reported_through_the_metric_hook(self):
        wrapped = rl.logged_reward_fn(correctness.make_reward_fn("vanilla"))
        wrapped(completions=["<answer>A</answer>"] * 4, ground_truth=["A"] * 4,
                sample_id=["p::original"] * 4)
        trainer = _FakeTrainer([wrapped])
        metrics = trainer.log_rollout_health()
        self.assertEqual(metrics["rollout_log/flat_ratio"], 1.0)
        self.assertEqual(metrics["rollout_log/groups"], 1.0)
        self.assertEqual(trainer.state.log_history[-1], metrics)

    def test_a_healthy_batch_reports_a_zero_flat_ratio(self):
        wrapped = rl.logged_reward_fn(correctness.make_reward_fn("vanilla"))
        wrapped(completions=["<answer>A</answer>", "<answer>B</answer>"],
                ground_truth=["A", "A"], sample_id=["p::original"] * 2)
        metrics = _FakeTrainer([wrapped]).log_rollout_health()
        self.assertEqual(metrics["rollout_log/flat_ratio"], 0.0)

    def test_invalid_outputs_are_counted(self):
        wrapped = rl.logged_reward_fn(correctness.make_reward_fn("vanilla"))
        wrapped(completions=["garbage", "garbage"], ground_truth=["A", "A"],
                sample_id=["p::original"] * 2)
        metrics = _FakeTrainer([wrapped]).log_rollout_health()
        self.assertEqual(metrics["rollout_log/invalid"], 2.0)
        self.assertEqual(metrics["rollout_log/underfilled"], 1.0)

    def test_no_reward_function_that_logs_is_not_an_error(self):
        trainer = _FakeTrainer([correctness.make_reward_fn("vanilla")])
        self.assertIsNone(trainer.log_rollout_health())

    def test_a_wrapped_model_module_is_unwrapped(self):
        """TRL wraps reward *models* in DDP; the attribute lives one level down."""
        wrapped = rl.logged_reward_fn(correctness.make_reward_fn("vanilla"))
        wrapped(completions=["<answer>A</answer>"], ground_truth=["A"],
                sample_id=["p::original"])

        class Module(object):
            pass

        holder = Module()
        holder.module = wrapped
        trainer = _FakeTrainer([holder])
        self.assertIsNotNone(trainer.log_rollout_health())


class ManifestDeclaresTheLogTest(unittest.TestCase):
    def test_write_manifest_records_the_rollout_log_block(self):
        config = config_mod.load_config(
            str(paths.CONFIGS_DIR / "vanilla.yaml"))
        directory = tempfile.mkdtemp(prefix="_manifest_", dir=str(paths.RUNS_DIR))
        try:
            # to_dict() also reports the derived budget numbers under "derived";
            # they are outputs of validation, not inputs, so they are dropped here.
            payload = dict(config.to_dict())
            payload.pop("derived", None)
            payload["output_dir"] = directory
            scoped = config_mod.validate_config(payload)
            target = grpo_base.write_manifest(scoped)
            manifest = json.loads(Path(target).read_text())
            block = manifest["rollout_log"]
            self.assertTrue(block["enabled"])
            self.assertEqual(block["module"], "rewards.rollout_log")
            self.assertEqual(block["metrics_prefix"], "rollout_log/")
            # The fields the brief asks to preserve must all be named.
            for field in ("prompt_id", "completion", "reward", "reason"):
                self.assertIn(field, block["per_rollout_fields"])
        finally:
            shutil.rmtree(directory, ignore_errors=True)


class AllMethodsLogRolloutsTest(unittest.TestCase):
    def test_every_method_wraps_its_reward_function(self):
        """A method whose reward fn is not wrapped silently loses Sec. 5.4."""
        from trainers.j1 import trl_trainer as j1_entry
        from trainers.j4r import trl_trainer as j4r_entry
        from trainers.jwc import trl_trainer as jwc_entry
        from trainers.vanilla import trl_trainer as vanilla_entry

        entries = {"vanilla": vanilla_entry, "j1": j1_entry,
                   "j4r": j4r_entry, "jwc": jwc_entry}
        files = {"vanilla": "vanilla.yaml", "j1": "j1.yaml",
                 "j4r": "j4r.yaml", "jwc": "jwc_rl_brier.yaml"}
        for method, entry in entries.items():
            config = config_mod.load_config(str(paths.CONFIGS_DIR / files[method]))
            funcs = entry.make_reward_funcs(config)
            for func in funcs:
                self.assertTrue(
                    hasattr(func, "last_rollouts"),
                    "{0}: reward function {1} does not keep a rollout log".format(
                        method, func.__name__))
                self.assertTrue(hasattr(func, "last_diagnostics"))


if __name__ == "__main__":
    unittest.main()
