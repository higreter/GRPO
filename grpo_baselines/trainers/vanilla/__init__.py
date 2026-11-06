"""Vanilla GRPO: TRL's trainer, unchanged, with a verdict-correctness reward.

The control condition.  Everything that distinguishes J1 (paired states + consistency),
J4R (EIS advantage + format reward) and JwC (probability head + Brier reward) is absent
here on purpose: this is the run those three are compared against.

Files
-----
``trl_trainer.py``
    Entry point (``python -m trainers.vanilla.trl_trainer``).  Its
    ``build_trainer_class`` returns ``None``, i.e. plain ``trl.GRPOTrainer``; the only
    shared piece it uses is ``SamplerMixin``, which pins the row order so a paired run
    and this one see the same pairs in the same sequence.

Reward
------
``rewards.correctness.make_reward_fn``: ``+1`` for the correct parsed verdict, ``0``
otherwise, with the parse failure recorded as a reason code rather than dropped.
"""

from __future__ import annotations

__all__ = ["trl_trainer"]
