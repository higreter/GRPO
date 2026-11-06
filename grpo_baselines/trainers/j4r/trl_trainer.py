"""Method 3 entry point: J4R (EIS advantage + judgement/format reward).

Launch (not executed in this task -- see the report's status table)::

    python -m trainers.j4r.trl_trainer --config configs/j4r.yaml --dry-run

What makes this J4R rather than GRPO with a different reward
------------------------------------------------------------
Two mechanisms, both required by arXiv:2505.13346:

* the **reward**, Eq. 6: ``R_j + R_f`` (:mod:`trainers.j4r.reward`), and
* the **advantage**, Eq. 4: EIS = global term + local term over the two equivalent
  states (:mod:`trainers.j4r.advantage`).

The second one is what a naive port drops.  TRL's own normalisation is per-group, and the
group here is one *state*, so using TRL unchanged would reproduce exactly the paper's
"separate groups" ablation while still being labelled J4R.  The advantage mixin is
therefore part of the trainer class, not an option, and :func:`build_trainer` refuses to
build a J4R trainer whose config does not ask for two states.
"""

from __future__ import annotations

from typing import Any, List, Optional

from common import config as config_mod
from rewards import rollout_log
from trainers import grpo_base

from . import reward as j4r_reward
from .advantage import EISAdvantageMixin

METHOD = "j4r"


def make_reward_funcs(config: config_mod.Config) -> List[Any]:
    """The single J4R reward function (Eq. 6 is already a sum)."""
    if config.states_per_sample != 2:
        raise ValueError(
            "J4R needs two equivalent states per sample (states_per_sample=2), got {0}; "
            "EIS would have nothing to integrate over.".format(config.states_per_sample)
        )
    # EIS normalises twice: globally across both states of a pair, and locally
    # within one state. The local group is exactly sample_id, so the default
    # grouping gives the diagnostic that matches the local term.
    return [rollout_log.logged_reward_fn(
        j4r_reward.make_reward_fn(config.output_format))]


def build_trainer_class(config: config_mod.Config) -> type:
    """``EISAdvantageMixin`` in front of TRL's ``GRPOTrainer``."""
    from trainers.trl_bridge import require

    trl = require("trl", "the J4R trainer")
    base = trl.GRPOTrainer

    class J4RTrainer(EISAdvantageMixin, base):
        """GRPO with J4R's EIS advantage replacing the per-state normalisation."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.eis_expected_states_per_pair = config.states_per_sample
            self.eis_expected_rollouts_per_state = config.rollouts_per_state

    J4RTrainer.__name__ = "J4RTrainer"
    return J4RTrainer


def build(trainer_class: Optional[type] = None):
    """Return the shared wiring's ``train`` bound to J4R's reward and trainer class."""
    return lambda config, **kwargs: grpo_base.train(
        config,
        reward_funcs=make_reward_funcs(config),
        trainer_class=trainer_class or build_trainer_class(config),
        **kwargs,
    )


def main(argv: Optional[List[str]] = None) -> int:
    return grpo_base.run_cli(
        __doc__ or "J4R trainer",
        reward_funcs_factory=make_reward_funcs,
        trainer_class_factory=build_trainer_class,
        argv=argv,
    )


__all__ = ["METHOD", "build", "build_trainer_class", "main", "make_reward_funcs"]


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
