"""The "writable" helpers must never turn a file path into a directory.

This bug happened twice: `write_jsonl`/`write_json` (fixed earlier) and then
`length_audit.write_audit` (found by running the documented command, which left a
directory named `runs/length_audit.json` behind and only failed at `open()`). Both times
the symptom pointed at the wrong line, because `mkdir` succeeds silently on a path that
was meant to be a file.

A test per call site is what stops the third one, so this walks every writer the project
ships rather than testing the helper in isolation.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path

from common import paths
from data import length_audit


class _Sandbox(object):
    """A throwaway directory inside the sanctioned writable root."""

    def __enter__(self):
        self.path = tempfile.mkdtemp(prefix="_writable_", dir=str(paths.RUNS_DIR))
        return Path(self.path)

    def __exit__(self, *exc):
        shutil.rmtree(self.path, ignore_errors=True)
        return False


class WritersNeverMkdirFilePathsTest(unittest.TestCase):
    def test_write_audit_creates_a_file_not_a_directory(self):
        with _Sandbox() as directory:
            target = directory / "audit.json"
            length_audit.write_audit(str(target), {"methods": {}})
            self.assertTrue(target.is_file())
            self.assertFalse(target.is_dir())

    def test_write_audit_overwrites_an_existing_file(self):
        with _Sandbox() as directory:
            target = directory / "audit.json"
            target.write_text("stale")
            length_audit.write_audit(str(target), {"methods": {"vanilla": {}}})
            self.assertIn("vanilla", target.read_text())

    def test_write_audit_refuses_a_path_that_is_already_a_directory(self):
        """The `mkdir`-a-file-path failure mode, asserted directly."""
        with _Sandbox() as directory:
            target = directory / "already_a_dir"
            target.mkdir()
            with self.assertRaises(IsADirectoryError):
                length_audit.write_audit(str(target), {"methods": {}})

    def test_write_json_and_jsonl_also_create_files(self):
        from common import reproducibility

        with _Sandbox() as directory:
            single = directory / "one.json"
            lines = directory / "many.jsonl"
            reproducibility.write_json(str(single), {"a": 1})
            reproducibility.write_jsonl(str(lines), [{"a": 1}, {"a": 2}])
            self.assertTrue(single.is_file())
            self.assertTrue(lines.is_file())
            self.assertEqual(len(lines.read_text().strip().splitlines()), 2)

    def test_the_audit_script_writes_where_it_says_it_does(self):
        """`scripts/audit_lengths.py` documents runs/length_audit.json as its default."""
        source = (paths.PROJECT_ROOT / "scripts" / "audit_lengths.py").read_text()
        self.assertIn("runs/length_audit.json", source)
        # and the module must be importable both as a script and as -m
        self.assertIn("sys.path", source,
                      "the script needs the project root on sys.path to run directly")


if __name__ == "__main__":
    unittest.main()
