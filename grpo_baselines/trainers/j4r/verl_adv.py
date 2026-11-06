"""J4R's EIS advantage as a verl advantage estimator.

verl keeps a registry of advantage estimators (``verl/trainer/ppo/core_algos.py:113-145``)
and dispatches to it from ``compute_advantage``
(``verl/trainer/ppo/ray_trainer.py:187-289``)::

    adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
    adv_kwargs = {"token_level_rewards": ..., "response_mask": ..., "config": ...}
    if "index" in inspect.signature(adv_estimator_fn).parameters:
        adv_kwargs["index"] = data.non_tensor_batch["uid"]
    advantages, returns = adv_estimator_fn(**adv_kwargs)

Two consequences shape this module:

1. **The only grouping key verl hands over is ``index``** (the ``uid`` non-tensor column).
   Both EIS levels must therefore be encoded in that one string.  We use
   ``"<pair_id>::<view>"`` -- the same ``sample_id`` the TRL path uses -- and derive the
   pair from the prefix.  A run whose ``uid`` values are plain ``"0"``/``"1"`` will fail
   loudly rather than silently degrade to per-group GRPO.
2. **The signature is the contract.**  It must accept ``token_level_rewards``,
   ``response_mask``, ``index`` and ``config`` by keyword, and return
   ``(advantages, returns)`` both shaped ``(bs, response_length)``.  ``config`` is accepted
   and ignored apart from an optional epsilon override, because everything EIS needs is in
   the index.

Numerical consistency with the TRL path.  Both integrations call
:func:`trainers.j4r.eis.compute_eis_advantages`, so the advantages are the same numbers.
That matters because the paper's worked example (Appendix D.1) is only reproduced with
**population** standard deviations; verl's own ``compute_grpo_outcome_advantage`` uses
``torch.std`` (sample std, ``correction=1``).  We keep the paper's convention and say so,
rather than matching the framework and quietly changing the estimator.

This module imports cleanly without verl installed: :func:`register` does the import, and
:func:`einsum_advantage` only needs torch.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from .advantage import compute_advantages, sample_id_parts
from .eis import EPSILON

#: Name registered in verl's ``ADV_ESTIMATOR_REGISTRY``; put it in
#: ``actor_rollout_ref.actor.adv_estimator`` (the YAML key at
#: ``verl/trainer/config/ppo_trainer.yaml:78``).
ADV_ESTIMATOR_NAME = "j4r_eis"

#: The ``uid`` convention this estimator requires, quoted by the error messages.
UID_CONVENTION = "<pair_id>::<view>"


def _uid_list(index: Any, count: int) -> List[str]:
    """Normalise verl's ``index`` argument to a list of strings."""
    if index is None:
        raise ValueError(
            "the EIS advantage estimator needs verl's `uid` column, which must look like "
            "'{0}'. Pass it as `data.non_tensor_batch['uid']`.".format(UID_CONVENTION)
        )
    values = [str(item) for item in list(index)]
    if len(values) != count:
        raise ValueError(
            "index has {0} entries but the batch has {1} rows".format(len(values), count)
        )
    return values


def _epsilon_from(config: Any) -> float:
    """Read an optional ``epsilon`` override from the verl ``AlgoConfig``."""
    if config is None:
        return EPSILON
    for path in (("epsilon",), ("eis_epsilon",), ("adv_estimator_epsilon",)):
        target = config
        found = True
        for attribute in path:
            target = getattr(target, attribute, None) if not isinstance(target, dict) else target.get(attribute)
            if target is None:
                found = False
                break
        if found:
            try:
                value = float(target)
            except (TypeError, ValueError):
                continue
            if value > 0.0:
                return value
    return EPSILON


def einsum_advantage(
    token_level_rewards: Any,
    response_mask: Any,
    index: Any = None,
    config: Any = None,
    **kwargs: Any,
) -> Tuple[Any, Any]:
    """verl-compatible EIS advantage estimator.

    ``token_level_rewards`` is ``(bs, response_length)``; the paper's ``R`` is the
    sequence-level return, which is what verl's own GRPO estimator uses as well
    (``scores = token_level_rewards.sum(dim=-1)``).  Advantages are broadcast over the
    response tokens and masked, matching ``compute_grpo_outcome_advantage``'s return
    convention so the actor's loss needs no special case.
    """
    del kwargs
    import torch

    if token_level_rewards.dim() != 2:
        raise ValueError(
            "token_level_rewards must be (bs, response_length), got shape {0}".format(
                tuple(token_level_rewards.shape))
        )
    if response_mask.shape != token_level_rewards.shape:
        raise ValueError(
            "response_mask shape {0} does not match token_level_rewards shape {1}".format(
                tuple(response_mask.shape), tuple(token_level_rewards.shape))
        )
    count = int(token_level_rewards.shape[0])
    scores = token_level_rewards.sum(dim=-1).detach().to("cpu").tolist()
    sample_ids = _uid_list(index, count)
    result = compute_advantages(scores, sample_ids, epsilon=_epsilon_from(config))
    values = result["advantages"]
    if len(values) != count:
        raise ValueError(
            "EIS produced {0} advantages for {1} rows".format(len(values), count)
        )
    advantages = torch.tensor(
        values, dtype=token_level_rewards.dtype, device=token_level_rewards.device
    )
    advantages = advantages.unsqueeze(-1) * response_mask
    # verl's contract: (advantages, returns).  GRPO-style estimators return the same
    # tensor twice (see compute_grpo_outcome_advantage's `return scores, scores`).
    return advantages, advantages.clone()


def register(core_algos: Optional[Any] = None) -> Any:
    """Register :func:`einsum_advantage` under :data:`ADV_ESTIMATOR_NAME`.

    Call before ``compute_advantage`` runs, e.g. from a verl entry point::

        from trainers.j4r.verl_adv import register
        register()

    Returns the registered function.  Registering twice with the same function is a
    no-op in verl; registering a *different* function under the same name raises there,
    which is the intended behaviour (it would mean two incompatible estimators).
    """
    if core_algos is None:
        from verl.trainer.ppo import core_algos  # noqa: F401 - imported for its side effect
    return core_algos.register_adv_est(ADV_ESTIMATOR_NAME)(einsum_advantage)


def estimator_config(adv_estimator: str = ADV_ESTIMATOR_NAME) -> Dict[str, str]:
    """The two config values a verl run needs, for the report and the launch scripts."""
    return {
        "adv_estimator": adv_estimator,
        "uid_convention": UID_CONVENTION,
    }


__all__ = [
    "ADV_ESTIMATOR_NAME",
    "UID_CONVENTION",
    "einsum_advantage",
    "estimator_config",
    "register",
]
