"""The upstream pins must describe the trees that are actually checked out.

The report claims the four read-only repositories were not modified. That claim is only
checkable if something compares the recorded commit against the real `git HEAD` and the
working tree against `git status`. Neither needs torch, TRL or a GPU, so it runs here.

If a checkout is absent (the project can be inspected without the read-only inputs mounted)
the test skips rather than failing: a missing input is not evidence of modification.
"""

from __future__ import annotations

import subprocess
import unittest

from common import paths


def _git(repo: str, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(paths.EXTERNAL_DIR / repo)] + list(args),
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "git {0} failed in {1}: {2}".format(
                " ".join(args), repo, result.stderr.strip() or result.stdout.strip()
            )
        )
    return result.stdout.strip()


class UpstreamPinTest(unittest.TestCase):
    """`common/paths.py:UPSTREAM_PINS` must match the trees on disk."""

    def test_the_pins_name_the_four_documented_repositories(self):
        self.assertEqual(
            sorted(paths.UPSTREAM_PINS),
            ["deepseek-math", "reasoning-judge-bench", "trl", "verl"],
        )

    def test_every_pin_has_a_url_and_a_full_commit_hash(self):
        for name, pin in paths.UPSTREAM_PINS.items():
            with self.subTest(repo=name):
                self.assertTrue(pin["url"].startswith("https://github.com/"))
                self.assertTrue(pin["url"].endswith(".git"))
                self.assertEqual(len(pin["commit"]), 40, pin["commit"])
                int(pin["commit"], 16)  # raises if it is not hexadecimal

    def test_each_recorded_commit_is_the_checked_out_head(self):
        for name, pin in paths.UPSTREAM_PINS.items():
            with self.subTest(repo=name):
                if not (paths.EXTERNAL_DIR / name / ".git").exists():
                    self.skipTest("{0} is not checked out here".format(name))
                self.assertEqual(_git(name, "rev-parse", "HEAD"), pin["commit"])

    def test_no_read_only_repository_has_local_modifications(self):
        for name in paths.UPSTREAM_PINS:
            with self.subTest(repo=name):
                if not (paths.EXTERNAL_DIR / name / ".git").exists():
                    self.skipTest("{0} is not checked out here".format(name))
                self.assertEqual(_git(name, "status", "--porcelain"), "")

    def test_no_read_only_repository_is_ahead_of_its_pin(self):
        for name, pin in paths.UPSTREAM_PINS.items():
            with self.subTest(repo=name):
                if not (paths.EXTERNAL_DIR / name / ".git").exists():
                    self.skipTest("{0} is not checked out here".format(name))
                # `git diff <pin> HEAD --stat` is empty exactly when HEAD is the pin and no
                # commit was added on top of it.
                self.assertEqual(_git(name, "diff", pin["commit"], "HEAD", "--stat"), "")


class ReadOnlyBoundaryTest(unittest.TestCase):
    """Every read-only input the brief names must actually be reachable and non-empty."""

    def test_the_required_runtime_inputs_exist(self):
        self.assertTrue(paths.HELPSTEER_TRAIN.is_file())
        self.assertTrue(paths.REWARDBENCH_OOD.is_file())
        self.assertTrue(paths.DATABASE_DIR.is_dir())
        self.assertTrue((paths.BASE_MODEL / "config.json").is_file())

    def test_the_four_paper_texts_are_present(self):
        if not paths.PAPERS_DIR.is_dir():
            self.skipTest("papers/ is intentionally optional in the migration bundle")
        found = sorted(p.name for p in paths.PAPERS_DIR.glob("*.txt"))
        for expected in (
            "deepseekmath_2402.03300.txt",
            "j1_2505.10320.txt",
            "j4r_2505.13346.txt",
            "judging_with_confidence_2510.00263.txt",
        ):
            with self.subTest(paper=expected):
                self.assertIn(expected, found)

    def test_writing_into_any_read_only_prefix_is_refused(self):
        for prefix in paths.READ_ONLY_PREFIXES:
            with self.subTest(prefix=str(prefix)):
                with self.assertRaises(PermissionError):
                    paths.assert_read_only_safe(prefix / "anything.txt")

    def test_the_model_directory_is_never_a_write_target(self):
        with self.assertRaises(PermissionError):
            paths.assert_read_only_safe(paths.BASE_MODEL / "pytorch_model.bin")


if __name__ == "__main__":
    unittest.main()
