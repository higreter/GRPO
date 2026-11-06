"""Shared, dependency-light building blocks for the four judge-GRPO baselines.

Importing this package must never pull in ``torch``/``transformers``: every module here
is usable from a bare Python 3.9+ interpreter so the CPU test-suite runs inside the
existing read-only ``judgelrm-sft`` environment without installing anything.
"""

from __future__ import annotations

__all__ = ["config", "labels", "parsing", "paths", "prompts", "reproducibility"]
