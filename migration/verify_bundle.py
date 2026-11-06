#!/usr/bin/env python3
"""Offline integrity and portability check for the GRPO transfer directory.

The base model and papers are intentionally outside the bundle.  Supplying
``--base-model`` additionally checks that the destination has a complete local
Qwen2.5-3B-Instruct snapshot, without loading weights or touching a GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PROJECT = ROOT / "grpo_baselines"

DATA = {
    "helpsteer3_train_raw.jsonl.gz": (None, "32b52e1d378f8dab1e4c9ae549da49a5d6fc0875aeafe3f9139e6053beb906bb"),
    "helpsteer3_test_raw.jsonl.gz": (None, "cd0d8b6efd7869a44c1b5ff91701232062555829b3c838abfd770d7dab6c7861"),
    "helpsteer3_train_clean.jsonl": (22190, "67784a938e36ed2375faada34b67406ab4befff8cc8b09bcff4613a24dd7af78"),
    "helpsteer3_test_clean.jsonl": (1866, "8ffb55e29fa1b73f0ab98f7c441f38bceed807b26a6faf5dff17fdcb2715f703"),
    "rewardbench_v1_ood_clean.jsonl": (2985, "9dc7b9e26fad89a0ac4483ed41df35747da6a576fa78d17330d164c0d7d7199c"),
}

REPOSITORIES = {
    "deepseek-math": "b8b0f8ce093d80bf8e9a641e44142f06d092c305",
    "reasoning-judge-bench": "e5a6c64fc97dd65f2b1b868664fb12c2d12b0606",
    "trl": "3885117d6a1dd647d43befeddf0da482c505cb47",
    "verl": "745432653d15af14954af1e055c8aeed72afa064",
}

MODEL_FILES = (
    "config.json",
    "generation_config.json",
    "merges.txt",
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)

REQUIRED = (
    "README.md",
    "RUNBOOK.md",
    "docs/IMPLEMENTATION.md",
    ".gitignore",
    ".gitattributes",
    "database/README.md",
    "grpo_baselines/common/paths.py",
    "grpo_baselines/configs/vanilla.yaml",
    "grpo_baselines/configs/j1.yaml",
    "grpo_baselines/configs/j4r.yaml",
    "grpo_baselines/configs/jwc_rl_brier.yaml",
    "grpo_baselines/trainers/dual_device_reference.py",
    "grpo_baselines/scripts/watch_and_launch_smokes.sh",
    "grpo_baselines/requirements/locks/grpo-trl-py311.lock.txt",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def line_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def git_head(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-model",
        default=os.environ.get("GRPO_BASE_MODEL"),
        help="external Qwen2.5-3B-Instruct directory (not included in bundle)",
    )
    parser.add_argument(
        "--runtime",
        action="store_true",
        help="also import and report the active Python training stack",
    )
    args = parser.parse_args()

    result: dict[str, object] = {
        "root": str(ROOT),
        "required": {},
        "data": {},
        "repositories": {},
        "base_model": {"included_in_bundle": False, "checked": bool(args.base_model)},
        "runtime": {"checked": args.runtime},
        "errors": [],
    }
    errors: list[str] = result["errors"]  # type: ignore[assignment]

    for relative in REQUIRED:
        exists = (ROOT / relative).is_file()
        result["required"][relative] = exists  # type: ignore[index]
        if not exists:
            errors.append("missing required file: " + relative)

    for name, (expected_lines, expected_hash) in DATA.items():
        path = ROOT / "database" / name
        item: dict[str, object] = {"exists": path.is_file()}
        if path.is_file():
            item["bytes"] = path.stat().st_size
            item["sha256"] = sha256(path)
            item["sha256_ok"] = item["sha256"] == expected_hash
            if expected_lines is not None:
                item["lines"] = line_count(path)
                item["lines_ok"] = item["lines"] == expected_lines
            if not item["sha256_ok"] or item.get("lines_ok") is False:
                errors.append("data verification failed: " + name)
        else:
            errors.append("missing data file: " + name)
        result["data"][name] = item  # type: ignore[index]

    for name, expected in REPOSITORIES.items():
        path = ROOT / "external" / name
        head = git_head(path)
        item = {"head": head, "expected": expected, "ok": head == expected}
        result["repositories"][name] = item  # type: ignore[index]
        if head != expected:
            errors.append("repository pin mismatch: " + name)

    if args.base_model:
        model = Path(args.base_model).expanduser().resolve()
        missing = [name for name in MODEL_FILES if not (model / name).is_file()]
        result["base_model"] = {
            "included_in_bundle": False,
            "checked": True,
            "path": str(model),
            "missing_files": missing,
            "ok": not missing,
        }
        if missing:
            errors.append("base model is incomplete: " + ", ".join(missing))

    if args.runtime:
        packages = {}
        for name in ("torch", "transformers", "tokenizers", "accelerate", "datasets", "trl", "yaml"):
            try:
                module = __import__(name)
                packages[name] = getattr(module, "__version__", "present")
            except Exception as exc:  # diagnostic only
                packages[name] = "ERROR: {0}: {1}".format(type(exc).__name__, exc)
                errors.append("runtime import failed: " + name)
        result["runtime"] = {
            "checked": True,
            "python": sys.version,
            "executable": sys.executable,
            "packages": packages,
        }

    result["ok"] = not errors
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
