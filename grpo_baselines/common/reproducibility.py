"""Deterministic seeding, run manifests and small IO helpers.

Everything here is standard-library only so that the CPU test-suite and the data
preparation steps run in the read-only ``judgelrm-sft`` environment.  ``torch`` is
imported lazily and only inside :func:`seed_torch`, which is never called by tests.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import sys
from typing import Any, Dict, Iterable, Iterator, Optional

from . import paths


def seed_everything(seed: int) -> int:
    """Seed the ``random``/``numpy`` (if importable) RNGs; return the seed."""
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an int")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:  # numpy is present in the audit env, but stay importable without it
        import numpy as np

        np.random.seed(seed)
    except Exception:  # pragma: no cover - numpy is optional for pure-logic tests
        pass
    return seed


def seed_torch(seed: int, *, deterministic: bool = True) -> int:
    """Seed torch/CUDA.  Imported lazily so the module stays CPU-testable."""
    seed_everything(seed)
    import torch  # noqa: WPS433 - deliberate lazy import

    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - requires a GPU
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed


def sha256_file(path: str, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def count_jsonl_rows(path: str) -> int:
    total = 0
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                total += 1
    return total


def iter_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError("{0}:{1}: invalid JSON ({2})".format(path, lineno, exc))
            if not isinstance(record, dict):
                raise ValueError("{0}:{1}: expected a JSON object".format(path, lineno))
            yield record


def write_jsonl(path: str, records: Iterable[Dict[str, Any]]) -> int:
    paths.assert_writable_file(path)
    total = 0
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
            total += 1
    return total


def write_json(path: str, payload: Any) -> None:
    paths.assert_writable_file(path)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def environment_snapshot(extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Record the interpreter/package state a run was produced under."""
    snapshot: Dict[str, Any] = {
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "upstream_pins": dict(paths.UPSTREAM_PINS),
        "base_model": str(paths.BASE_MODEL),
        "project_root": str(paths.PROJECT_ROOT),
    }
    try:
        import torch

        snapshot["torch"] = torch.__version__
        snapshot["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():  # pragma: no cover - requires a GPU
            snapshot["cuda_device_count"] = int(torch.cuda.device_count())
    except Exception as exc:  # pragma: no cover
        snapshot["torch"] = "unavailable: {0}".format(exc)
    for name in ("transformers", "trl", "verl", "vllm", "ray"):
        try:
            module = __import__(name)
            snapshot[name] = getattr(module, "__version__", "unknown")
        except Exception:
            snapshot[name] = None
    if extra:
        snapshot.update(extra)
    return snapshot


__all__ = [
    "count_jsonl_rows",
    "environment_snapshot",
    "iter_jsonl",
    "seed_everything",
    "seed_torch",
    "sha256_file",
    "sha256_text",
    "write_json",
    "write_jsonl",
]
