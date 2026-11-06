"""J1: paired-view GRPO with verdict correctness + consistency (arXiv:2505.10320).

Modules
-------
``trl_trainer.py``
    Entry point (``python -m trainers.j1.trl_trainer``).

The reward lives in ``rewards/j1.py`` rather than here, unlike J4R's, because it is a
pure function of two parsed verdicts and two ground truths -- it does not depend on J1's
prompt layout or on any trainer mechanism.  What *does* depend on the trainer is the
assumption that both views of a pair arrive in the same generation batch, and that is
what ``J1ConsistencyGuardMixin`` checks.

The consistency term is not "both letters equal"
------------------------------------------------
Because the second view swaps the answers, the correct letter is *reversed* between the
two views.  So "did the model answer the same letter twice" would measure the position
bias J1 is trying to remove.  ``rewards.j1.score_pair`` maps both verdicts back to the
physical answer (:func:`common.labels.display_to_response`) and only then compares, and
``same_response_chosen`` is reported separately from ``both_correct``: a judge can be
perfectly position-consistent and still wrong, and collapsing the two numbers would hide
exactly the failure mode this method targets.
"""

from __future__ import annotations

__all__ = ["trl_trainer"]
