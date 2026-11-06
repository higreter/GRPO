"""Check every number in the report that the code can recompute.

Prose drifts: the "Four thousand and ninety-six sequences per update" sentence survived
several editing passes because nothing compared it to the config. This walks the claims
that have a single source of truth and fails when they disagree.

Only *recomputable* claims are checked -- test counts, budget quantities, token limits,
the erratum value, the census. Sentences that are judgement calls ("EIS is not vanilla
normalisation") are left to the tests.

Two mechanical notes, both learned the hard way:

* The report writes large numbers with thin grouping ("16 384", "46 268 416"), so
  comparison strips spaces and commas from both sides before matching.
* The exact-test-count check is deliberately a fixed point: the report must quote the
  number this suite actually contains. Adding tests without updating the report makes
  this test fail, which is the point -- a count nobody maintains is worse than no count.
"""
from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT.parent / "docs" / "IMPLEMENTATION.md"

sys.path.insert(0, str(ROOT))

from common import config as config_mod  # noqa: E402
from common import paths  # noqa: E402
from data import rewardbench_adapter  # noqa: E402
from trainers.j4r import eis  # noqa: E402


def _squash(text):
    """Drop the grouping characters so '16 384' and '16384' compare equal."""
    return re.sub(r"[,\u2009\u202f\s]", "", text)


class ReportMatchesTheCodeTest(unittest.TestCase):
    def setUp(self):
        self.raw = REPORT.read_text(encoding="utf-8")
        self.flat = _squash(self.raw)
        self.configs = config_mod.load_all_configs()

    def test_the_test_count_quoted_is_the_one_discovery_finds(self):
        loader = unittest.TestLoader()
        suite = loader.discover(start_dir=str(ROOT / "tests"), top_level_dir=str(ROOT))
        discovered = suite.countTestCases()
        quoted = {int(m) for m in re.findall(r"Ran(\d+)tests", self.flat)}
        quoted |= {int(m) for m in re.findall(r"\((\d+)tests\)", self.flat)}
        self.assertIn(
            discovered, quoted,
            "report quotes {0}; discovery finds {1}. Update the report.".format(
                sorted(quoted), discovered))

    def test_total_rollouts_and_tokens_are_quoted_correctly(self):
        for method, cfg in self.configs.items():
            self.assertIn(str(cfg.total_rollouts_derived), self.flat,
                          "{0}: total_rollouts not quoted".format(method))
            self.assertIn(str(cfg["estimated_generated_tokens"]), self.flat,
                          "{0}: generated tokens not quoted".format(method))

    def test_the_sequence_bound_quoted_matches_every_config(self):
        for method, cfg in self.configs.items():
            bound = cfg.max_prompt_length + cfg.max_completion_length
            self.assertLessEqual(bound, paths.MAX_CONTEXT_TOKENS, method)
            self.assertIn(str(bound), self.flat,
                          "{0}: the prompt+completion bound is not quoted".format(method))

    def test_the_paper_erratum_number_is_the_computed_one(self):
        """0.285 must appear where the D.1 correction is explained."""
        self.assertAlmostEqual(eis.PAPER_EXAMPLE_GLOBAL_ONLY[("B", 1.0)], 0.285, places=4)
        self.assertIn("0.285", self.raw)
        self.assertIn("2.017", self.raw)

    def test_the_subset_census_quoted_matches_the_file(self):
        scan = rewardbench_adapter.scan_source()
        self.assertEqual(scan["rows"], 2985)
        self.assertEqual(scan["subsets"], 23)
        self.assertEqual(scan["unknown_subsets"], [])
        self.assertIn("2985", self.flat)
        self.assertIn("23officials", self.flat)
        for subset in rewardbench_adapter.OFFICIAL_SUBSETS:
            if subset.startswith("xstest"):
                self.assertIn(subset, self.raw,
                              "{0} is missing from the report".format(subset))

    def test_no_stale_counts_survive(self):
        """Old numbers that were true once are the most misleading kind."""
        for needle in ("338 tests", "290 tests", "237 tests", "Ran348"):
            self.assertNotIn(needle, self.flat,
                             "report still quotes the stale count {0!r}".format(needle))

    def test_the_false_sequence_count_is_gone(self):
        """'4096 sequences per update' was never true; an update is 64 or 128."""
        self.assertNotIn("Fourthousandandninety-six", self.flat)
        self.assertNotIn("4096sequences", self.flat)
        for method, cfg in self.configs.items():
            per_update = cfg.rollouts_per_update
            self.assertIn(per_update, (64, 128), method)

    def test_the_per_file_table_matches_discovery_row_by_row(self):
        """The table's rows must be the modules that exist, with the counts they have."""
        loader = unittest.TestLoader()
        measured = {}
        for path in sorted((ROOT / "tests").glob("test_*.py")):
            measured[path.name] = loader.loadTestsFromName(
                "tests." + path.stem).countTestCases()

        rows = re.findall(r"^\| `(test_[a-z0-9_]+\.py)` \| (\d+) \|", self.raw, re.MULTILINE)
        self.assertTrue(rows, "the per-file table is missing from the report")

        quoted = dict((name, int(count)) for name, count in rows)
        self.assertEqual(len(quoted), len(rows),
                         "the table lists the same file twice: {0}".format(sorted(quoted)))
        self.assertEqual(sorted(quoted), sorted(measured),
                         "the table and the tests/ directory disagree about which files "
                         "exist; table={0} measured={1}".format(
                             sorted(quoted), sorted(measured)))
        for name, count in sorted(measured.items()):
            self.assertEqual(quoted[name], count,
                             "{0}: report says {1} tests, discovery finds {2}".format(
                                 name, quoted[name], count))
        self.assertEqual(sum(quoted.values()), sum(measured.values()))

    def test_the_file_tree_matches_the_directory(self):
        """Every module in the tree, and no module that is not there.

        The tree is regenerated by `/tmp/yh_grpo/py/regen_tree.py`; this test is what
        makes forgetting to run that script a failure instead of a stale section.
        """
        block = re.search(r"```\ncommon/\{.*?```", self.raw, re.DOTALL)
        self.assertIsNotNone(block, "the Sec. 6 file tree is missing")
        tree = block.group(0)

        for group in ("common", "data", "rewards", "trainers", "evaluation", "tests"):
            actual = sorted(p.stem for p in (ROOT / group).glob("*.py"))
            self.assertTrue(actual, group)
            for stem in actual:
                self.assertIn(stem, tree,
                              "{0}/{1}.py exists but is not in the tree".format(group, stem))

        for method in ("vanilla", "j1", "j4r", "jwc"):
            for path in (ROOT / "trainers" / method).glob("*.py"):
                self.assertIn(path.stem, tree,
                              "trainers/{0}/{1}.py is not in the tree".format(
                                  method, path.stem))

        for path in (ROOT / "configs").glob("*.yaml"):
            self.assertIn(path.stem, tree,
                          "configs/{0}.yaml is not in the tree".format(path.stem))

        for path in (ROOT / "scripts").iterdir():
            if path.is_file():
                self.assertIn(path.name, tree, "scripts/{0} is not in the tree".format(path.name))

    def test_the_status_flags_all_say_not_run(self):
        for phrase in ("GPU smoke test | **Not run**",
                       "Full training | **Not run**",
                       "RewardBench inference | **Not run**"):
            self.assertIn(phrase, self.raw)

    def test_the_report_names_all_four_methods_and_their_entries(self):
        for method, entry in (("Vanilla", "trainers/vanilla/"),
                              ("J1", "trainers/j1/"),
                              ("J4R", "trainers/j4r/"),
                              ("JwC", "trainers/jwc/")):
            self.assertIn(method, self.raw)
            self.assertIn(entry, self.raw)


if __name__ == "__main__":
    unittest.main()
