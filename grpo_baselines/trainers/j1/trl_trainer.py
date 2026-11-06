"""Method 2 entry point: J1 (verdict correctness + pair-level consistency).

Launch (not executed in this task -- see the report's status table)::

    python -m trainers.j1.trl_trainer --config configs/j1.yaml --dry-run

J1 in one paragraph
-------------------
For a preference pair ``(x, a, b)`` the model is asked twice, once as ``(x, a, b)`` and
once as ``(x, b, a)``, and must emit ``<think>...</think><answer>[[A]]</answer>``.  The
reward has exactly two terms (arXiv:2505.10320 Sec. 2.2): ``+1`` when the verdict is
correct, and ``+1`` more only when *both* orders are correct.  There is no format bonus --
the paper says they tried one and it did not help -- so a malformed ``<think>`` block is
recorded but not penalised.

Why this needs a guard the other methods do not
-----------------------------------------------
The consistency term is a property of a *pair*, but TRL's advantage is computed per
prompt-state.  ``rewards.j1.make_reward_fn`` bridges that by regrouping the batch on
``sample_id`` and broadcasting the ``+1`` to both rows of a pair.  That bridge is only
correct while both views of every pair are present in the same generation batch; if one
were missing, the reward would silently degrade to plain correctness and the run would
still be labelled J1.  :class:`J1ConsistencyGuardMixin` therefore turns a missing
counterpart into an error instead of a quieter experiment.

What is *not* here
------------------
J1 uses TRL's ordinary per-state advantage.  The paired states are separate prompts with
their own labels (``A`` in the original view, ``B`` in the swapped one); nothing is
integrated across them.  That distinction is exactly what separates J1 from J4R.
"""

from __future__ import annotations

from typing import Any, List, Optional

from common import config as config_mod
from rewards import j1 as j1_reward, rollout_log
from trainers import grpo_base

METHOD = "j1"


class J1BatchError(RuntimeError):
    """A generation batch could not be scored as complete pairs."""


def _pair_stats(reward_funcs: List[Any]) -> Optional[dict]:
    for func in reward_funcs:
        stats = getattr(func, "last_stats", None)
        if stats:
            return stats
    return None


class J1ConsistencyGuardMixin(object):
    """Fail loudly when a generation batch does not contain both views of every pair."""

    def _generate_and_score_completions(self, inputs):
        output = super()._generate_and_score_completions(inputs)
        if not getattr(self.model, "training", True):
            return output
        stats = _pair_stats(list(getattr(self, "reward_funcs", []) or []))
        if stats is None:
            raise J1BatchError(
                "the J1 reward function did not record pair statistics; the consistency "
                "term cannot be verified for this batch"
            )
        unpaired = int(stats.get("unpaired_groups", 0))
        if unpaired:
            raise J1BatchError(
                "{0} of {1} groups had no counterpart in the generation batch. J1's "
                "consistency term needs the original and swapped views in the same batch; "
                "with the counterpart missing the reward silently degrades to plain "
                "correctness.".format(unpaired, int(stats.get("paired_groups", 0)) + unpaired)
            )
        return output


def make_reward_funcs(config: config_mod.Config) -> List[Any]:
    """J1's single reward function (correctness + pair consistency)."""
    if config.states_per_sample != 2:
        raise ValueError(
            "J1 needs the original and swapped views of every pair (states_per_sample=2), "
            "got {0}; the consistency term would have no counterpart.".format(
                config.states_per_sample)
        )
    # The rollout log groups by sample_id, i.e. by state: that is the group TRL
    # normalises within, so the flat-group diagnostic describes the same group the
    # advantage uses.
    return [rollout_log.logged_reward_fn(
        j1_reward.make_reward_fn(config.output_format))]


def build_trainer_class(config: config_mod.Config) -> type:
    """``J1ConsistencyGuardMixin`` in front of TRL's ``GRPOTrainer``."""
    from trainers.trl_bridge import require

    trl = require("trl", "the J1 trainer")
    base = trl.GRPOTrainer

    class J1Trainer(J1ConsistencyGuardMixin, base):
        """GRPO with J1's pair-level consistency reward."""

    J1Trainer.__name__ = "J1Trainer"
    del config
    return J1Trainer


def build(trainer_class: Optional[type] = None):
    """Return the shared wiring's ``train`` bound to J1's reward and trainer class."""
    return lambda config, **kwargs: grpo_base.train(
        config,
        reward_funcs=make_reward_funcs(config),
        trainer_class=trainer_class or build_trainer_class(config),
        **kwargs,
    )


def main(argv: Optional[List[str]] = None) -> int:
    return grpo_base.run_cli(
        __doc__ or "J1 trainer",
        reward_funcs_factory=make_reward_funcs,
        trainer_class_factory=build_trainer_class,
        argv=argv,
    )


__all__ = [
    "J1BatchError",
    "J1ConsistencyGuardMixin",
    "METHOD",
    "build",
    "build_trainer_class",
    "main",
    "make_reward_funcs",
]


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
