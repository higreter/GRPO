"""Tests for CLI argument validation and resume semantics (Sec. 9)."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from common import paths
from trainers import grpo_base


def _parse(argv):
    return grpo_base.build_arg_parser("test").parse_args(argv)


class LimitValidationTest(unittest.TestCase):
    def test_a_limit_of_zero_is_refused(self):
        with self.assertRaises(grpo_base.ArgumentError) as caught:
            grpo_base.validate_args(_parse(["--config", "x", "--limit", "0"]))
        self.assertIn("--limit must be >= 1", str(caught.exception))

    def test_a_negative_limit_is_refused(self):
        with self.assertRaises(grpo_base.ArgumentError):
            grpo_base.validate_args(_parse(["--config", "x", "--limit", "-3"]))

    def test_a_positive_limit_passes(self):
        grpo_base.validate_args(_parse(["--config", "x", "--limit", "4"]))

    def test_no_limit_passes(self):
        grpo_base.validate_args(_parse(["--config", "x"]))


class ResumeValidationTest(unittest.TestCase):
    def test_a_missing_checkpoint_directory_is_refused(self):
        with self.assertRaises(grpo_base.ArgumentError) as caught:
            grpo_base.validate_args(
                _parse(["--config", "x", "--resume-from-checkpoint", "/nope/nope"]))
        self.assertIn("does not exist", str(caught.exception))

    def test_a_file_instead_of_a_directory_is_refused(self):
        handle = tempfile.NamedTemporaryFile(
            prefix="_ckpt_", suffix=".json", dir=str(paths.RUNS_DIR), delete=False)
        handle.write(b"{}")
        handle.close()
        try:
            with self.assertRaises(grpo_base.ArgumentError) as caught:
                grpo_base.validate_args(
                    _parse(["--config", "x", "--resume-from-checkpoint", handle.name]))
            self.assertIn("not a directory", str(caught.exception))
        finally:
            Path(handle.name).unlink(missing_ok=True)

    def test_a_directory_without_trainer_state_is_refused(self):
        directory = tempfile.mkdtemp(prefix="_ckpt_", dir=str(paths.RUNS_DIR))
        try:
            with self.assertRaises(grpo_base.ArgumentError) as caught:
                grpo_base.validate_args(
                    _parse(["--config", "x", "--resume-from-checkpoint", directory]))
            self.assertIn("trainer_state.json", str(caught.exception))
        finally:
            Path(directory).rmdir()

    def test_a_real_checkpoint_directory_passes(self):
        directory = tempfile.mkdtemp(prefix="_ckpt_", dir=str(paths.RUNS_DIR))
        try:
            (Path(directory) / "trainer_state.json").write_text("{}")
            (Path(directory) / "optimizer.pt").write_bytes(b"test placeholder")
            grpo_base.validate_args(
                _parse(["--config", "x", "--resume-from-checkpoint", directory]))
        finally:
            (Path(directory) / "trainer_state.json").unlink()
            (Path(directory) / "optimizer.pt").unlink()
            Path(directory).rmdir()


class OutputDirSemanticsTest(unittest.TestCase):
    def test_every_shipped_config_names_an_output_dir_inside_the_project(self):
        for config in _shipped():
            target = Path(config["output_dir"])
            if not target.is_absolute():
                target = paths.PROJECT_ROOT / target
            self.assertTrue(
                paths.is_inside(target.resolve(), paths.PROJECT_ROOT),
                "{0}: output_dir {1} escapes the project root".format(
                    config.method, target))


class DryRunResumeNoteTest(unittest.TestCase):
    def test_dry_run_still_validates_the_checkpoint_path(self):
        """Ignoring the checkpoint on a dry run must not mean skipping its validation."""
        with self.assertRaises(grpo_base.ArgumentError):
            grpo_base.validate_args(
                _parse(["--config", "x", "--dry-run",
                        "--resume-from-checkpoint", "/nope/nope"]))


def _shipped():
    from common import config as config_mod
    return list(config_mod.load_all_configs().values())


if __name__ == "__main__":
    unittest.main()
