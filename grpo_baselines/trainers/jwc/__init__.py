"""JwC: GRPO with an RL-Brier reward over a verbalised probability (arXiv:2510.00263).

Modules
-------
``trl_trainer.py``
    Entry point (``python -m trainers.jwc.trl_trainer``).

The reward lives in ``rewards/jwc_brier.py``: it is a function of one parsed probability
and one binary target, with no dependency on the prompt layout or on any trainer
mechanism.  Both the Brier form (Eq. 1, used for training) and the log form (Eq. 2, kept
for the report's comparison and for ``epsilon=1e-3`` clipping) are implemented there, and
``common.parsing.parse_jwc`` is the strict finite-number parser they rely on -- ``NaN``,
``inf``, out-of-range, duplicated or missing tags all become a recorded reason code.

Evaluating this method
----------------------
The probability means "B is better".  ``common.labels.probability_target`` maps a ground
truth to ``1``/``0`` accordingly, and ``evaluation.metrics`` reports the Brier score
(lower is better -- note the reward is ``1 - brier``), ECE and MCE for this method only.
``calibration_metrics`` refuses the other three methods rather than assuming their hard
verdict is a confidence of 1.0, which would make their calibration numbers meaningless
and flatter the calibrated one.
"""

from __future__ import annotations

__all__ = ["trl_trainer"]
