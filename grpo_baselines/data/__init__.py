"""Dataset adapters and prompt-state construction.

Everything here reads the read-only ``database/`` files and produces the minimal training
/ evaluation schema; nothing in this package imports torch or transformers at module level.
"""

from __future__ import annotations

__all__ = ["helpsteer_adapter", "length_audit", "pair_views", "rewardbench_adapter"]
