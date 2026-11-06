"""Pin the eight Sec. 9 facts per method, so the table cannot quietly lose one.

The budget table is duplicated documentation: `common/config.py` owns the truth for the
arithmetic and `tests/test_configs.py` already checks the four derived quantities. What was
*not* checked is the set of questions Sec. 9 enumerates -- in particular the token columns
and whether each method runs a frozen reference with KL.

That omission is the kind that survives review: the shared `controlled:` block had
`kl_beta`, so the table looked complete, and a per-method KL of 0 would have gone unnoticed.
"""
from __future__ import annotations

import unittest

import yaml

from common import config as config_mod
from common import paths


class BudgetTableAnswersSec9Test(unittest.TestCase):
    REQUIRED = (
        "states_per_sample",
        "rollouts_per_state",
        "rollouts_per_update",
        "training_steps",
        "estimated_generated_tokens",
        "actual_generated_tokens",
        "reference_model_policy",
        "kl_beta",
        "frozen_reference_with_kl",
    )

    def setUp(self):
        self.table = yaml.safe_load(
            (paths.CONFIGS_DIR / "controlled_budget.yaml").read_text(encoding="utf-8"))
        self.configs = config_mod.load_all_configs()

    def test_every_method_row_answers_all_eight_questions(self):
        for method in self.configs:
            row = self.table["methods"][method]
            missing = [key for key in self.REQUIRED if key not in row]
            self.assertEqual(missing, [], "{0} row is missing {1}".format(method, missing))

    def test_the_rows_agree_with_the_shipped_configs(self):
        for method, cfg in self.configs.items():
            row = self.table["methods"][method]
            self.assertEqual(row["states_per_sample"], cfg.states_per_sample, method)
            self.assertEqual(row["rollouts_per_state"], cfg.rollouts_per_state, method)
            self.assertEqual(row["rollouts_per_update"], cfg.rollouts_per_update, method)
            self.assertEqual(row["training_steps"], cfg.training_steps, method)
            self.assertEqual(row["estimated_generated_tokens"],
                             cfg["estimated_generated_tokens"], method)
            self.assertEqual(row["kl_beta"], cfg.kl_beta, method)
            self.assertEqual(row["reference_model_policy"],
                             cfg["reference_model_policy"], method)

    def test_actual_tokens_are_null_and_say_why(self):
        """A measured token count cannot exist when nothing was trained."""
        for method, row in self.table["methods"].items():
            self.assertIsNone(row["actual_generated_tokens"],
                              "{0} claims a measured token count".format(method))
        self.assertIn("open_risks", self.table)
        joined = " ".join(str(item) for item in self.table["open_risks"])
        self.assertIn("No GPU smoke test was run", joined)

    def test_every_method_row_states_the_reference_policy_explicitly(self):
        """KL must be visible per method, not inferred from the shared block."""
        for method, row in self.table["methods"].items():
            self.assertIn(row["reference_model_policy"],
                          (config_mod.REFERENCE_POLICY_FROZEN,
                           config_mod.REFERENCE_POLICY_NONE), method)
            if row["reference_model_policy"] == config_mod.REFERENCE_POLICY_FROZEN:
                self.assertGreater(row["kl_beta"], 0.0, method)
                self.assertTrue(row["frozen_reference_with_kl"], method)
            else:
                self.assertEqual(row["kl_beta"], 0.0, method)
                self.assertFalse(row["frozen_reference_with_kl"], method)


if __name__ == "__main__":
    unittest.main()
