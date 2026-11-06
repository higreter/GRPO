"""Method-specific trainers.

Layout
------
``trainers/trl_bridge.py``
    Shared, import-light glue between a validated :class:`common.config.Config`
    and TRL's ``GRPOConfig`` / ``GRPOTrainer``.  Nothing here imports ``trl``,
    ``torch`` or ``datasets`` at module import time, so this package stays
    importable in the read-only conda environment used for the CPU test suite.

``trainers/vanilla/``
    Method 1: standard GRPO on single-state prompts with a verdict-correctness
    reward (TRL ``GRPOTrainer`` used as-is).

``trainers/j1/``
    Method 2: J1's verdict correctness + pair-level consistency, implemented on
    top of TRL with a paired sampler and a reward function that regroups rows by
    ``pair_id``.

``trainers/j4r/``
    Method 3: J4R's Equivalent-State Integration (EIS) advantage.  ``eis.py`` and
    ``advantage.py`` are pure Python (torch-free) so the exact advantage values
    are unit-testable; ``trl_trainer.py`` and ``verl_adv.py`` are the two
    framework integrations.

``trainers/jwc/``
    Method 4: JwC's RL-Brier reward over a verbalised probability.

Anything that needs a deep-learning stack imports it inside a function, and
raises :class:`trainers.trl_bridge.MissingDependencyError` with the exact install
instructions from ``requirements/`` when it is unavailable.  That is deliberate:
this repository must be inspectable and testable without a GPU.
"""
