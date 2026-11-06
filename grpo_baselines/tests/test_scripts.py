"""The launch scripts must reject bad arguments *before* doing anything expensive.

Sec. 9 of the brief ends with: "所有脚本须有参数校验、明确输出目录和可恢复 checkpoint 语义".
A guard that only runs after the conda env is activated is not a guard on this host -- the
env does not exist, so `set -e` aborts first and the user sees a bare exit 1 with no
message. These tests therefore assert two separate things:

* a bad argument exits 2, with the reason on stderr;
* a *valid* argument gets past the guard and stops at exit 4, the named
  `GRPO_CONFIRM_TRAIN` opt-in, which is the proof that the guard did not reject something
  it should have accepted.

Nothing here trains, loads a model, or creates a checkpoint: the scripts stop at the opt-in
gate, which is exactly the boundary this task is allowed to reach.

The opt-in gate exists because of a real failure.  An earlier version of this file asserted
exit 3 ("conda env does not exist") as proof that the arguments were accepted -- which made
the test pass only while the environment was missing.  When `grpo-trl-py311` was created,
the same test ran the script to completion: it loaded Qwen2.5-3B and started a training
step before running out of GPU memory.  The exit code now comes from a variable the caller
controls, so the assertion no longer encodes the state of the machine.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from common import paths

SCRIPTS = paths.PROJECT_ROOT / "scripts"
METHODS = ("vanilla", "j1", "j4r", "jwc")


def run(script, *args):
    """Run a launch script with a clean environment and capture its exit + output."""
    environment = dict(os.environ)
    environment.pop("CUDA_VISIBLE_DEVICES", None)
    completed = subprocess.run(
        ["bash", str(SCRIPTS / script), *args],
        cwd=str(paths.PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
        timeout=60,
    )
    return completed.returncode, completed.stdout.decode(), completed.stderr.decode()


class TrainScriptValidationTest(unittest.TestCase):
    def test_an_unknown_flag_is_rejected_before_the_environment_is_touched(self):
        for method in METHODS:
            code, _out, err = run("train_{0}.sh".format(method), "--wat")
            self.assertEqual(code, 2, "{0} accepted --wat".format(method))
            self.assertIn("usage:", err)
            self.assertNotIn("conda env", err,
                             "{0} reached the env check, so the guard ran too late".format(
                                 method))

    def test_a_missing_checkpoint_is_rejected(self):
        missing = "/tmp/_no_such_checkpoint_for_test"
        self.assertFalse(Path(missing).exists())
        for method in METHODS:
            code, _out, err = run("train_{0}.sh".format(method),
                                  "--resume-from-checkpoint", missing)
            self.assertEqual(code, 2)
            self.assertIn("is not a directory", err)

    def test_a_checkpoint_without_trainer_state_is_rejected(self):
        directory = tempfile.mkdtemp(prefix="_ckpt_no_state_")
        try:
            Path(directory, "optimizer.pt").touch()
            for method in METHODS:
                code, _out, err = run("train_{0}.sh".format(method),
                                      "--resume-from-checkpoint", directory)
                self.assertEqual(code, 2)
                self.assertIn("trainer_state.json", err)
        finally:
            Path(directory, "optimizer.pt").unlink()
            os.rmdir(directory)

    def test_a_model_only_checkpoint_without_optimizer_state_is_rejected(self):
        directory = tempfile.mkdtemp(prefix="_ckpt_no_optimizer_")
        try:
            Path(directory, "trainer_state.json").touch()
            for method in METHODS:
                code, _out, err = run("train_{0}.sh".format(method),
                                      "--resume-from-checkpoint", directory)
                self.assertEqual(code, 2)
                self.assertIn("optimizer.pt", err)
        finally:
            Path(directory, "trainer_state.json").unlink()
            os.rmdir(directory)

    def test_a_resume_flag_without_a_value_is_rejected(self):
        for method in METHODS:
            code, _out, err = run("train_{0}.sh".format(method),
                                  "--resume-from-checkpoint")
            self.assertEqual(code, 2)
            self.assertIn("usage:", err)

    def test_a_valid_invocation_gets_past_the_guard(self):
        """Exit 4 is the opt-in gate, which proves the arguments were accepted.

        The assertion is on the code the *gate* produces, not on what happens to be
        installed: an exit of 4 means the argument guard let a well-formed command
        through, and that nothing downstream of it ran.
        """
        before = sorted(os.listdir(str(paths.RUNS_DIR)))
        for method in METHODS:
            code, out, err = run("train_{0}.sh".format(method))
            self.assertEqual(code, 4, "{0}: expected the opt-in gate".format(method))
            self.assertIn("config:", out)
            self.assertIn("output:", out)
            self.assertIn("GRPO_CONFIRM_TRAIN=yes", err)
        # And the point of the gate: a bare invocation wrote no run artefacts.
        self.assertEqual(before, sorted(os.listdir(str(paths.RUNS_DIR))),
                         "a bare launch script wrote into runs/")

    def test_the_printed_output_dir_matches_the_config(self):
        """A redirected run must not announce the old destination."""
        import yaml

        for method, filename in (("vanilla", "vanilla.yaml"), ("j1", "j1.yaml"),
                                 ("j4r", "j4r.yaml"), ("jwc", "jwc_rl_brier.yaml")):
            config = yaml.safe_load((paths.CONFIGS_DIR / filename).read_text())
            code, out, _err = run("train_{0}.sh".format(method))
            self.assertEqual(code, 4)
            announced = [line for line in out.splitlines()
                         if line.startswith("output:")]
            self.assertEqual(len(announced), 1)
            self.assertIn(config["output_dir"], announced[0],
                          "{0} announced a different dir than its config".format(method))


class EvalScriptValidationTest(unittest.TestCase):
    def test_an_unknown_method_is_rejected(self):
        code, _out, err = run("eval_rewardbench.sh", "j2", "/tmp")
        self.assertEqual(code, 2)
        self.assertIn("METHOD must be one of", err)

    def test_a_missing_checkpoint_is_rejected(self):
        code, _out, err = run("eval_rewardbench.sh", "vanilla", "/tmp/_no_ckpt_here")
        self.assertEqual(code, 2)
        self.assertIn("does not exist", err)

    def test_a_directory_that_is_not_a_checkpoint_is_rejected(self):
        directory = tempfile.mkdtemp(prefix="_not_a_checkpoint_")
        try:
            code, _out, err = run("eval_rewardbench.sh", "vanilla", directory)
            self.assertEqual(code, 2)
            self.assertIn("config.json", err)
        finally:
            os.rmdir(directory)

    def test_missing_positional_arguments_are_rejected(self):
        code, _out, err = run("eval_rewardbench.sh")
        self.assertEqual(code, 1)  # `${1:?}` under `set -e`
        self.assertIn("usage", err)


if __name__ == "__main__":
    unittest.main()
