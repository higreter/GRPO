"""RewardBench evaluation package.

``metrics`` holds the metric definitions and is pure standard library, so every number in
the report can be recomputed on a CPU-only machine from a saved predictions file.
``rewardbench`` holds the runner that produces those predictions; it is the only module in
the project that needs a GPU, and it is never executed as part of the implementation task.
"""

from __future__ import annotations

__all__ = ["metrics", "rewardbench"]
