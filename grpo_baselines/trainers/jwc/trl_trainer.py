"""Method 4 entry point: JwC (GRPO with an RL-Brier reward over a verbalised probability).

Launch (not executed in this task -- see the report's status table)::

    python -m trainers.jwc.trl_trainer --config configs/jwc_rl_brier.yaml --dry-run

JwC in one paragraph
--------------------
The model does not pick A or B; it states ``p = Pr[B better]`` in a
``<prob_rb_better>0.85</prob_rb_better>`` tag and the reward is the Brier score of that
probability against the hard label (arXiv:2510.00263 Eq. 1)::

    R(r; y) = 1 - (p - y)^2,    y = 1 if B is better else 0

An output whose probability cannot be parsed gets ``0`` (the paper's unparsable branch),
not a neutral ``1 - (0.5 - y)^2`` -- otherwise refusing to answer would be weakly
rewarded.

The one substantive deviation, stated up front
----------------------------------------------
The paper trains on *dense* preference distributions: ten annotators per prompt, so ``y``
is a rate in ``[0, 1]``.  HelpSteer3 supplies one label per pair, so ``y`` here is
``0`` or ``1`` and the objective is the sparse-label instance of the same estimator.
This is not a detail to bury: the Brier reward is proper for both, but a
single-label target carries less information, and the report says so.

What the trainer does *not* do
------------------------------
No temperature scaling, no threshold tuning, no calibration fitting.  ``p`` is the model's
own verbalised probability; :mod:`evaluation.metrics` reports Brier/ECE/MCE on it as-is.
Fitting a post-hoc calibrator and then reporting a calibrated Brier next to the other
methods' hard verdicts would compare a two-parameter fit against a raw score.
"""

from __future__ import annotations

from typing import Any, List, Optional

from common import config as config_mod
from rewards import jwc_brier, rollout_log
from trainers import grpo_base

METHOD = "jwc"


def make_reward_funcs(config: config_mod.Config) -> List[Any]:
    """RL-Brier reward on the verbalised probability."""
    if config.states_per_sample != 1:
        raise ValueError(
            "JwC uses one state per sample (the probability head already carries the "
            "uncertainty), got states_per_sample={0}.".format(config.states_per_sample)
        )
    return [rollout_log.logged_reward_fn(
        jwc_brier.make_reward_fn(config.output_format))]


def build_trainer_class(config: config_mod.Config) -> Optional[type]:
    """``None`` -- TRL's ``GRPOTrainer`` as-is, plus the shared paired sampler."""
    del config
    return None


def build(trainer_class: Optional[type] = None):
    """Return the shared wiring's ``train`` bound to JwC's reward."""
    return lambda config, **kwargs: grpo_base.train(
        config,
        reward_funcs=make_reward_funcs(config),
        trainer_class=trainer_class,
        **kwargs,
    )


def main(argv: Optional[List[str]] = None) -> int:
    return grpo_base.run_cli(
        __doc__ or "JwC trainer",
        reward_funcs_factory=make_reward_funcs,
        trainer_class_factory=build_trainer_class,
        argv=argv,
    )


__all__ = ["METHOD", "build", "build_trainer_class", "main", "make_reward_funcs"]


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
