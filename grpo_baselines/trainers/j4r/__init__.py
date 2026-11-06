"""J4R: GRPO with the Equivalent-State Integration advantage (arXiv:2505.13346).

Modules
-------
``eis.py``
    The estimator itself, and nothing else: given ``(pair, state, rollout, reward)``
    records it returns ``(R - mean_all)/std_all + (R - mean_state)/std_state``.  Pure
    Python, so the paper's worked example (Appendix D.1) can be checked numerically on a
    CPU.  Note the erratum recorded there: the paper's "global advantage only" table
    prints the *local* value for subgroup B's reward-1.0 row; the global term is 0.285.

``reward.py``
    Eq. 6: judgement reward (correct verdict) plus a ``+/-0.5`` format reward.  It lives
    here rather than in ``rewards/`` because the format term is defined on J4R's
    ``Explanation:`` / ``Verdict: [[A>B]]`` layout, which is also what fixes the two
    equivalent states EIS integrates over.  It also records ``last_rewards`` so the
    advantage mixin can read exactly the scalars TRL grouped on.

``advantage.py``
    ``EISAdvantageMixin``: lets TRL generate, tokenise, and compute the clipped surrogate
    loss, then replaces ``advantages`` with EIS values.  Without this, TRL's per-group
    normalisation *is* the paper's "separate groups" ablation.

``trl_trainer.py``
    Entry point for the TRL integration (``python -m trainers.j4r.trl_trainer``).

``verl_adv.py``
    Entry point for the verl integration: the same estimator registered as
    ``j4r_eis``.  Both integrations call ``eis.compute_eis_advantages``, so they cannot
    drift apart numerically.

Two states are mandatory
------------------------
``states_per_sample`` must be 2 for J4R: the ``original`` and ``swapped`` views of one
preference pair.  A single-state J4R run would have no second state to integrate over,
so the loader and the config validator both reject it rather than falling back to GRPO.
"""

from __future__ import annotations

__all__ = ["advantage", "eis", "reward", "trl_trainer", "verl_adv"]
