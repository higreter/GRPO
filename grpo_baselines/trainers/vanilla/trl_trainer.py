"""Method 1 entry point: vanilla GRPO with a verdict-correctness reward.

Launch (not executed in this task -- see the report's status table)::

    python -m trainers.vanilla.trl_trainer --config configs/vanilla.yaml --dry-run

This is the control condition the other three are measured against: TRL's
``GRPOTrainer`` unchanged (its own per-group normalisation, no paired states, no
calibration head, no EIS), with the only method-specific piece being the reward.

Why the reward is "verdict correct" and not something richer
-----------------------------------------------------------
The comparison in the report is about *advantage construction and reward design*, so the
control has to be the plainest thing that still trains: one prompt, one sample, ``+1`` if
the parsed verdict matches the label.  Adding a format bonus here would make the control
reward a different shape from J1's and would confound the comparison the paper makes.

The one non-obvious requirement is that the reward *reports* invalid outputs rather than
hiding them.  ``rewards.correctness.make_reward_fn`` returns ``0.0`` for an unparsable
completion (so it is scored, not dropped) and records the reason code; the trainer logs
the invalid rate, because a run whose invalid rate is high is not comparable to one whose
is low even at the same accuracy.
"""

from __future__ import annotations

from typing import Any, List, Optional

from common import config as config_mod
from rewards import correctness, rollout_log
from trainers import grpo_base

METHOD = "vanilla"


def make_reward_funcs(config: config_mod.Config) -> List[Any]:
    """Verdict correctness only, on the method's own output format."""
    if config.states_per_sample != 1:
        raise ValueError(
            "vanilla GRPO uses one state per sample, got states_per_sample={0}; the "
            "paired-states methods are j1/j4r.".format(config.states_per_sample)
        )
    return [rollout_log.logged_reward_fn(
        correctness.make_reward_fn(config.output_format))]


def build_trainer_class(config: config_mod.Config) -> Optional[type]:
    """``None`` -- TRL's ``GRPOTrainer`` as-is, plus the shared paired sampler."""
    del config
    return None


def build(trainer_class: Optional[type] = None):
    """Return the shared wiring's ``train`` bound to vanilla's reward."""
    return lambda config, **kwargs: grpo_base.train(
        config,
        reward_funcs=make_reward_funcs(config),
        trainer_class=trainer_class,
        **kwargs,
    )


def main(argv: Optional[List[str]] = None) -> int:
    return grpo_base.run_cli(
        __doc__ or "vanilla GRPO trainer",
        reward_funcs_factory=make_reward_funcs,
        trainer_class_factory=build_trainer_class,
        argv=argv,
    )


__all__ = ["METHOD", "build", "build_trainer_class", "main", "make_reward_funcs"]


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
