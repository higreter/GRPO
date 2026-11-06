"""Inject J4R's EIS advantage into TRL's GRPO loop.

The reward function records the exact scalars and row identities TRL scored. This
mixin replaces TRL's independent per-state normalisation with pair/global EIS,
before gradient accumulation splits the generation batch.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence, Tuple
from .eis import EPSILON, Rollout, compute_eis_advantages

class EISAdvantageError(RuntimeError):
    """The EIS advantage could not be computed for this batch."""

def sample_id_parts(sample_id: str) -> Tuple[str, str]:
    pair_id, sep, view = str(sample_id).partition("::")
    if not sep or not pair_id or not view:
        raise EISAdvantageError("sample_id {0!r} is not '<pair_id>::<view>'".format(sample_id))
    return pair_id, view

def rollouts_from_batch(rewards: Sequence[float], sample_ids: Sequence[str]) -> List[Rollout]:
    if len(rewards) != len(sample_ids):
        raise EISAdvantageError("rewards and sample_ids must have equal length")
    counters: Dict[Tuple[str, str], int] = {}
    rollouts: List[Rollout] = []
    for reward, sample_id in zip(rewards, sample_ids):
        pair_id, state_id = sample_id_parts(sample_id)
        key = (pair_id, state_id)
        index = counters.get(key, 0)
        counters[key] = index + 1
        rollouts.append(Rollout(pair_id, state_id, index, float(reward)))
    return rollouts

def compute_advantages(rewards: Sequence[float], sample_ids: Sequence[str], *,
    expected_states_per_pair: Optional[int] = None,
    expected_rollouts_per_state: Optional[int] = None,
    epsilon: float = EPSILON) -> Dict[str, Any]:
    result = compute_eis_advantages(rollouts_from_batch(rewards, sample_ids), epsilon=epsilon,
        expected_states_per_pair=expected_states_per_pair,
        expected_rollouts_per_state=expected_rollouts_per_state)
    result["sample_ids"] = [str(item) for item in sample_ids]
    return result

def _collect_sample_ids(inputs: Sequence[Dict[str, Any]]) -> Optional[List[str]]:
    if not inputs or "sample_id" not in inputs[0]:
        return None
    return [str(row["sample_id"]) for row in inputs]

def _unwrap_reward_func(func: Any) -> Any:
    for attribute in ("module", "_orig_mod"):
        wrapped = getattr(func, attribute, None)
        if wrapped is not None and callable(wrapped):
            return wrapped
    return func

class EISAdvantageMixin(object):
    """Replace TRL per-state normalisation; mixin must precede GRPOTrainer."""
    eis_expected_states_per_pair: Optional[int] = None
    eis_expected_rollouts_per_state: Optional[int] = None
    eis_last_result: Optional[Dict[str, Any]] = None

    def _eis_reward_values(self, count: int) -> Optional[List[float]]:
        for func in list(getattr(self, "reward_funcs", []) or []):
            candidate = _unwrap_reward_func(func)
            values = getattr(candidate, "last_rewards", None)
            if values is None:
                continue
            values = list(values)
            if len(values) != count:
                raise EISAdvantageError("recorded rewards do not align with generation rows")
            return [float(value) for value in values]
        return None

    def _eis_sample_ids(self, inputs: Sequence[Dict[str, Any]], count: int) -> List[str]:
        sample_ids = _collect_sample_ids(inputs)
        if sample_ids is None:
            raise EISAdvantageError("EIS requires sample_id on every generation row")
        if len(sample_ids) != count:
            raise EISAdvantageError("sample_id count does not align with generation rows")
        return sample_ids

    def _apply_eis_advantages(self, output: Dict[str, Any],
                              inputs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        import torch
        advantages = output.get("advantages")
        if advantages is None:
            raise EISAdvantageError("TRL returned no advantages entry to replace")
        count = int(advantages.shape[0])
        rewards = self._eis_reward_values(count)
        if rewards is None:
            raise EISAdvantageError("no reward function recorded per-row rewards")
        result = compute_advantages(rewards, self._eis_sample_ids(inputs, count),
            expected_states_per_pair=self.eis_expected_states_per_pair,
            expected_rollouts_per_state=self.eis_expected_rollouts_per_state)
        values = result["advantages"]
        if len(values) != count:
            raise EISAdvantageError("EIS advantage count does not match generation rows")
        output["advantages"] = torch.tensor(values, dtype=advantages.dtype,
                                             device=advantages.device)
        self.eis_last_result = result
        return output

    def _generate_and_score_completions(self, inputs):
        output = super()._generate_and_score_completions(inputs)
        return self._apply_eis_advantages(output, inputs)

__all__ = ["EISAdvantageError", "EISAdvantageMixin", "compute_advantages",
           "rollouts_from_batch", "sample_id_parts"]
