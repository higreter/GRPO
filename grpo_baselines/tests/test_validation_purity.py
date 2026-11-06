"""Config validation must be read-only on the filesystem.

`runs/_test_vanilla` was the evidence: a test validated a stub config and the directory
appeared. A validation that creates things is not a validation -- it also masks typos,
because a mistyped `output_dir` silently springs into existence and the run later writes
somewhere nobody looked.

These tests assert the absence of a side effect, which is the kind of property that is
easy to lose again in a refactor and invisible until it matters.

Note on the payload: `load_config` returns a `Config` (read-only accessors), and its
`to_dict()` carries a `derived` block that `validate_config` correctly rejects as an
unknown field. So the stub is built from the yaml document itself, which is the same thing
`validate_config` expects from any caller.
"""
from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import yaml

from common import config as config_mod
from common import paths


class ValidationCreatesNothingTest(unittest.TestCase):
    def setUp(self):
        self.payload = yaml.safe_load(
            (paths.CONFIGS_DIR / "vanilla.yaml").read_text(encoding="utf-8"))

    def _stub(self, output_dir):
        payload = copy.deepcopy(self.payload)
        payload["output_dir"] = str(output_dir)
        return payload

    def test_validating_a_config_does_not_create_the_output_dir(self):
        with tempfile.TemporaryDirectory(dir=str(paths.RUNS_DIR)) as directory:
            target = Path(directory) / "not_created_yet"
            config_mod.validate_config(self._stub(target))
            self.assertFalse(target.exists(),
                             "validate_config created the output directory")

    def test_validating_twice_still_creates_nothing(self):
        with tempfile.TemporaryDirectory(dir=str(paths.RUNS_DIR)) as directory:
            target = Path(directory) / "still_not_created"
            config_mod.validate_config(self._stub(target))
            config_mod.validate_config(self._stub(target))
            self.assertFalse(target.exists())

    def test_an_output_path_that_is_a_file_is_reported_as_such(self):
        with tempfile.TemporaryDirectory(dir=str(paths.RUNS_DIR)) as directory:
            blocker = Path(directory) / "occupied"
            blocker.write_text("i am a file")
            with self.assertRaises(NotADirectoryError):
                config_mod.validate_config(self._stub(blocker))

    def test_a_path_outside_the_project_is_still_refused(self):
        with self.assertRaises(PermissionError):
            config_mod.validate_config(self._stub("/tmp/definitely_not_allowed"))

    def test_every_shipped_config_validates_without_touching_the_filesystem(self):
        before = sorted(p for p in paths.RUNS_DIR.iterdir())
        configs = config_mod.load_all_configs()
        self.assertEqual(len(configs), 4)
        after = sorted(p for p in paths.RUNS_DIR.iterdir())
        self.assertEqual(before, after,
                         "loading the shipped configs changed runs/")

    def test_check_writable_output_is_the_pure_one(self):
        with tempfile.TemporaryDirectory(dir=str(paths.RUNS_DIR)) as directory:
            target = Path(directory) / "pure"
            returned = paths.check_writable_output(target)
            self.assertEqual(Path(returned), Path(target))
            self.assertFalse(target.exists())
            # ... whereas the creating variant does create
            created = paths.assert_writable_output(target)
            self.assertTrue(Path(created).is_dir())


if __name__ == "__main__":
    unittest.main()
