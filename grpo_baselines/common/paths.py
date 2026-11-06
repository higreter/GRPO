"""Single source of truth for every filesystem path used by the project.

Nothing under ``READ_ONLY_PREFIXES`` may ever be written to.  ``assert_writable`` guards
output directories at script start-up so a typo cannot silently overwrite the frozen
datasets or the pinned upstream repositories.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
GRPO_ROOT: Final[Path] = PROJECT_ROOT.parent

# --- read-only inputs -------------------------------------------------------------
DATABASE_DIR: Final[Path] = GRPO_ROOT / "database"
EXTERNAL_DIR: Final[Path] = GRPO_ROOT / "external"
PAPERS_DIR: Final[Path] = GRPO_ROOT / "papers"

HELPSTEER_TRAIN: Final[Path] = DATABASE_DIR / "helpsteer3_train_clean.jsonl"
HELPSTEER_TEST: Final[Path] = DATABASE_DIR / "helpsteer3_test_clean.jsonl"
REWARDBENCH_OOD: Final[Path] = DATABASE_DIR / "rewardbench_v1_ood_clean.jsonl"

DEFAULT_BASE_MODEL: Final[Path] = Path("/data/common_data/Qwen/Qwen2.5-3B-Instruct")
BASE_MODEL: Final[Path] = Path(
    os.environ.get("GRPO_BASE_MODEL", str(DEFAULT_BASE_MODEL))
).expanduser()
CONDA_ENV_PREFIX: Final[Path] = Path(
    os.environ.get(
        "GRPO_READ_ONLY_ENV",
        "/data1/yanghao/miniconda3/envs/judgelrm-sft",
    )
).expanduser()

READ_ONLY_PREFIXES: Final[tuple[Path, ...]] = (
    DATABASE_DIR,
    EXTERNAL_DIR,
    PAPERS_DIR,
    BASE_MODEL,
    CONDA_ENV_PREFIX,
)

# --- writable outputs -------------------------------------------------------------
CONFIGS_DIR: Final[Path] = PROJECT_ROOT / "configs"
ARTIFACTS_DIR: Final[Path] = PROJECT_ROOT / "artifacts"
RUNS_DIR: Final[Path] = PROJECT_ROOT / "runs"

UPSTREAM_PINS: Final[dict[str, dict[str, str]]] = {
    "deepseek-math": {
        "url": "https://github.com/deepseek-ai/DeepSeek-Math.git",
        "commit": "b8b0f8ce093d80bf8e9a641e44142f06d092c305",
    },
    "trl": {
        "url": "https://github.com/huggingface/trl.git",
        "commit": "3885117d6a1dd647d43befeddf0da482c505cb47",
    },
    "verl": {
        "url": "https://github.com/volcengine/verl.git",
        "commit": "745432653d15af14954af1e055c8aeed72afa064",
    },
    "reasoning-judge-bench": {
        "url": "https://github.com/SalesforceAIResearch/ReasoningJudgeBench.git",
        "commit": "e5a6c64fc97dd65f2b1b868664fb12c2d12b0606",
    },
}

MAX_CONTEXT_TOKENS: Final[int] = 32768


def _resolved(path: Path) -> Path:
    return Path(os.path.abspath(os.path.expanduser(str(path))))


def is_inside(path: Path, prefix: Path) -> bool:
    """True when ``path`` is ``prefix`` itself or lives below it (symlinks resolved)."""
    p = _resolved(path)
    root = _resolved(prefix)
    try:
        p.relative_to(root)
    except ValueError:
        return False
    return True


def assert_read_only_safe(target: Path) -> None:
    """Raise if ``target`` resolves inside any frozen input tree."""
    for prefix in READ_ONLY_PREFIXES:
        if is_inside(target, prefix):
            raise PermissionError(
                f"refusing to write to {target}: it resolves inside read-only input {prefix}"
            )


def _assert_inside_project(resolved: Path, kind: str) -> Path:
    if not is_inside(resolved, PROJECT_ROOT):
        raise PermissionError(
            f"{kind} {resolved} is outside the sanctioned project root {PROJECT_ROOT}"
        )
    assert_read_only_safe(resolved)
    return resolved


def check_writable_output(target: Path) -> Path:
    """Validate an output **directory** path without creating it.

    Used by config validation, which must be a pure check: creating `runs/<method>` just
    because someone loaded a yaml is a side effect nobody asked for, and it hides typos
    (the directory springs into existence and the run later writes into the wrong place).
    """
    resolved = _assert_inside_project(_resolved(target), "output directory")
    if resolved.exists() and not resolved.is_dir():
        raise NotADirectoryError(
            f"output directory {resolved} already exists as a file"
        )
    return resolved


def assert_writable_output(target: Path) -> Path:
    """Validate an output **directory** and create it.

    Only paths inside :data:`PROJECT_ROOT` are accepted; this is the machine-checkable
    form of "the only writable root is ``grpo_baselines/``".
    """
    resolved = _assert_inside_project(_resolved(target), "output directory")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def assert_writable_file(target: Path) -> Path:
    """Validate an output **file** path and create its parent directory.

    Separate from :func:`assert_writable_output` because calling ``mkdir`` on a file path
    would silently create a directory where the caller expects a JSONL file.
    """
    resolved = _assert_inside_project(_resolved(target), "output file")
    if resolved.exists() and resolved.is_dir():
        raise IsADirectoryError(
            f"output file {resolved} already exists as a directory"
        )
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


__all__ = [
    "ARTIFACTS_DIR",
    "BASE_MODEL",
    "DEFAULT_BASE_MODEL",
    "CONDA_ENV_PREFIX",
    "CONFIGS_DIR",
    "DATABASE_DIR",
    "EXTERNAL_DIR",
    "GRPO_ROOT",
    "HELPSTEER_TEST",
    "HELPSTEER_TRAIN",
    "MAX_CONTEXT_TOKENS",
    "PAPERS_DIR",
    "PROJECT_ROOT",
    "READ_ONLY_PREFIXES",
    "REWARDBENCH_OOD",
    "RUNS_DIR",
    "UPSTREAM_PINS",
    "assert_read_only_safe",
    "assert_writable_file",
    "assert_writable_output",
    "check_writable_output",
    "is_inside",
]
