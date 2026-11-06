"""EIS -- Equivalent-Input-Sampling advantage (J4R, arXiv:2505.13346 Sec. 3.2, Eq. 4).

Given the ``L`` equivalent prompt states ``T_1(q) .. T_L(q)`` of one source pair and the
rollouts sampled from each, the advantage of rollout ``i`` in state ``l`` is

    A^(i,l) = (R^(i,l) - mean_all) / std_all      <- global term
            + (R^(i,l) - mean_l)   / std_l        <- local term

**Both terms are required.**  The paper's Appendix D.1 ablates the global term and reports
JudgeBench dropping from 56.86 to 49.14; the local term alone is plain per-state GRPO and
loses the "these states are the same question" signal, while the global term alone
collapses the states into one pool and loses the per-state baseline.

Statistics are **population** standard deviations (divide by ``N``), which is what
reproduces the paper's worked example to three decimals -- see
:data:`PAPER_EXAMPLE` and ``tests/test_j4r_advantages.py``, which reproduces all twelve of
its numbers elementwise:

* state A = 12 x 1.5, 4 x 1.0 ; state B = 12 x 0.0, 4 x 1.0
* separate groups (vanilla) : +0.577 / -1.732 / +1.732 / -0.577
* global only               : +1.044 / +0.285 / +0.285 / -1.234
* global + local (EIS)      : +1.621 / -1.447 / +2.017 / -1.811

Erratum in the paper.  Appendix D.1 prints ``+1.732`` for the reward-1.0 rollouts *of
subgroup B* under "Global advantage only".  That is the subgroup-B **local** term: the
global mean is 26/32 = 0.8125 and the global population standard deviation is 0.6585, so
``(1.0 - 0.8125) / 0.6585 = 0.285`` for every reward-1.0 rollout regardless of subgroup.
The same paragraph's EIS value confirms it: ``0.285 + 1.732 = 2.017``, exactly the number
printed for EIS.  :data:`PAPER_EXAMPLE_GLOBAL_ONLY` therefore stores the arithmetically
consistent 0.285 and the test asserts on it; the printed 1.732 is quoted in the report as
an erratum rather than reproduced.

Design notes
------------
* The three identifier layers (pair -> state -> rollout) are distinct fields on
  :class:`Rollout`, not a concatenated string, so a mislabelled rollout fails a check
  instead of silently joining the wrong group.
* Zero variance is an *expected* condition (a pair all of whose rollouts are correct), so
  it is recorded, not crashed on: a zero-std term contributes 0, which is exactly
  ``(R - mean)/std`` in the limit since the numerator is 0 as well.  TRL's own
  ``1e-4`` floor achieves the same thing; :data:`EPSILON` is explicit here so the value is
  part of the contract.
* A state with a single rollout cannot be normalised locally, so it is rejected loudly.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

#: Denominator floor used when a term's standard deviation is (numerically) zero.
EPSILON = 1e-6

#: Tolerance for the "these two floats are the same to 3 dp" checks in the report.
PAPER_TOLERANCE = 5e-4


class EISError(ValueError):
    """Raised when the rollout structure cannot support EIS."""


class Rollout(object):
    """One sampled completion, identified at all three levels."""

    __slots__ = ("pair_id", "state_id", "rollout_index", "reward")

    def __init__(self, pair_id: str, state_id: str, rollout_index: int, reward: float) -> None:
        if not isinstance(pair_id, str) or not pair_id:
            raise EISError("pair_id must be a non-empty str")
        if not isinstance(state_id, str) or not state_id:
            raise EISError("state_id must be a non-empty str")
        if isinstance(rollout_index, bool) or not isinstance(rollout_index, int):
            raise EISError("rollout_index must be an int")
        if isinstance(reward, bool) or not isinstance(reward, (int, float)):
            raise EISError("reward must be a number, got {0!r}".format(type(reward).__name__))
        value = float(reward)
        if not math.isfinite(value):
            raise EISError("reward must be finite, got {0!r}".format(reward))
        self.pair_id = pair_id
        self.state_id = state_id
        self.rollout_index = rollout_index
        self.reward = value

    def key(self) -> Tuple[str, str, int]:
        return (self.pair_id, self.state_id, self.rollout_index)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "Rollout({0!r}, {1!r}, {2}, reward={3})".format(
            self.pair_id, self.state_id, self.rollout_index, self.reward)


def _mean(values: Sequence[float]) -> float:
    return math.fsum(values) / len(values)


def _pstd(values: Sequence[float]) -> float:
    """Population standard deviation (divide by N), matching the paper's example."""
    if len(values) < 2:
        return 0.0
    mu = _mean(values)
    return math.sqrt(math.fsum((value - mu) ** 2 for value in values) / len(values))


def _normalise(value: float, values: Sequence[float], epsilon: float) -> Tuple[float, bool]:
    """``(value - mean) / std`` with an explicit zero-variance branch.

    Returns ``(term, zero_variance_flag)``.  When the standard deviation is below
    ``epsilon`` the term is 0.0 -- and, for an exact zero variance, the numerator is
    provably 0 as well, so this is the limiting value rather than a fudge.
    """
    mu = _mean(values)
    sigma = _pstd(values)
    if sigma < epsilon:
        return 0.0, True
    return (value - mu) / sigma, False


def group_rollouts(rollouts: Sequence[Rollout]) -> Dict[str, Dict[str, List[Rollout]]]:
    """``pair_id -> state_id -> rollouts`` (states sorted later by the caller)."""
    grouped: Dict[str, Dict[str, List[Rollout]]] = {}
    for rollout in rollouts:
        grouped.setdefault(rollout.pair_id, {}).setdefault(rollout.state_id, []).append(rollout)
    return grouped


def compute_eis_advantages(
    rollouts: Sequence[Rollout],
    *,
    epsilon: float = EPSILON,
    expected_states_per_pair: Optional[int] = None,
    expected_rollouts_per_state: Optional[int] = None,
) -> Dict[str, Any]:
    """Compute EIS advantages aligned with the input order.

    Returns a dict with ``advantages`` (list of floats, same order as ``rollouts``),
    ``components`` (per-rollout global/local breakdown) and ``pairs`` (per-pair statistics:
    means, stds, zero-variance flags).  All diagnostics are part of the return value so the
    trainer can log them and the report can quote them.
    """
    if not rollouts:
        raise EISError("no rollouts supplied")
    if epsilon <= 0.0:
        raise EISError("epsilon must be positive")

    seen = set()
    for rollout in rollouts:
        key = rollout.key()
        if key in seen:
            raise EISError("duplicate rollout key {0!r}".format(key))
        seen.add(key)

    grouped = group_rollouts(rollouts)
    pair_stats: Dict[str, Any] = {}

    for pair_id in sorted(grouped):
        states = grouped[pair_id]
        state_count = len(states)
        if state_count < 2:
            raise EISError(
                "pair {0!r} has {1} state(s); EIS needs at least 2 equivalent states "
                "(T_1(q)..T_L(q))".format(pair_id, state_count)
            )
        if expected_states_per_pair is not None and state_count != expected_states_per_pair:
            raise EISError(
                "pair {0!r} has {1} states, expected {2}".format(
                    pair_id, state_count, expected_states_per_pair)
            )
        for state_id in sorted(states):
            count = len(states[state_id])
            if count < 2:
                raise EISError(
                    "pair {0!r} state {1!r} has {2} rollout(s); the local term needs >= 2 "
                    "to define a standard deviation".format(pair_id, state_id, count)
                )
            if expected_rollouts_per_state is not None and count != expected_rollouts_per_state:
                raise EISError(
                    "pair {0!r} state {1!r} has {2} rollouts, expected {3}".format(
                        pair_id, state_id, count, expected_rollouts_per_state)
                )
        # Rollout indexes identify a completion *within* its state, so they restart at 0
        # for every state; the three-level key (pair, state, index) is what is unique.
        for state_id, members in states.items():
            indexes = [r.rollout_index for r in members]
            if len(set(indexes)) != len(indexes):
                raise EISError(
                    "pair {0!r} state {1!r} has duplicate rollout_index values: {2}".format(
                        pair_id, state_id, sorted(indexes))
                )

    # -- global statistics are computed over every rollout of the pair ----------------
    global_values: Dict[str, List[float]] = {
        pair_id: [r.reward for state in states.values() for r in state]
        for pair_id, states in grouped.items()
    }

    advantages: List[float] = []
    components: List[Dict[str, Any]] = []
    for rollout in rollouts:
        states = grouped[rollout.pair_id]
        state_values = [r.reward for r in states[rollout.state_id]]
        global_term, global_zero = _normalise(
            rollout.reward, global_values[rollout.pair_id], epsilon)
        local_term, local_zero = _normalise(rollout.reward, state_values, epsilon)
        advantages.append(global_term + local_term)
        components.append(
            {
                "pair_id": rollout.pair_id,
                "state_id": rollout.state_id,
                "rollout_index": rollout.rollout_index,
                "reward": rollout.reward,
                "global_term": global_term,
                "local_term": local_term,
                "global_zero_variance": global_zero,
                "local_zero_variance": local_zero,
            }
        )

    for pair_id in sorted(grouped):
        states = grouped[pair_id]
        per_state = {}
        for state_id in sorted(states):
            values = [r.reward for r in states[state_id]]
            per_state[state_id] = {
                "count": len(values),
                "mean": _mean(values),
                "std": _pstd(values),
                "zero_variance": _pstd(values) < epsilon,
                "min": min(values),
                "max": max(values),
            }
        pair_stats[pair_id] = {
            "states": len(states),
            "rollouts": len(global_values[pair_id]),
            "global_mean": _mean(global_values[pair_id]),
            "global_std": _pstd(global_values[pair_id]),
            "global_zero_variance": _pstd(global_values[pair_id]) < epsilon,
            "per_state": per_state,
        }

    return {
        "advantages": advantages,
        "components": components,
        "pairs": pair_stats,
        "epsilon": epsilon,
        "num_rollouts": len(rollouts),
        "num_pairs": len(grouped),
        "zero_variance_global_pairs": sum(
            1 for stats in pair_stats.values() if stats["global_zero_variance"]),
        "zero_variance_state_count": sum(
            1 for stats in pair_stats.values() for s in stats["per_state"].values()
            if s["zero_variance"]),
    }


def advantages_by_key(result: Dict[str, Any]) -> Dict[Tuple[str, str, int], float]:
    """``(pair_id, state_id, rollout_index) -> advantage`` from a result dict."""
    return {
        (item["pair_id"], item["state_id"], item["rollout_index"]): value
        for item, value in zip(result["components"], result["advantages"])
    }


# --------------------------------------------------------------------------------------
# Paper's worked example, exposed so tests and the report quote one source of truth.
# --------------------------------------------------------------------------------------
def _paper_example_rollouts() -> List[Rollout]:
    rollouts: List[Rollout] = []
    for index in range(12):
        rollouts.append(Rollout("demo", "A", index, 1.5))
    for index in range(4):
        rollouts.append(Rollout("demo", "A", 12 + index, 1.0))
    for index in range(12):
        rollouts.append(Rollout("demo", "B", index, 0.0))
    for index in range(4):
        rollouts.append(Rollout("demo", "B", 12 + index, 1.0))
    return rollouts


PAPER_EXAMPLE = {
    # (state_id, reward) -> expected advantage, rounded to the paper's 3 decimals.
    ("A", 1.5): 1.621,
    ("A", 1.0): -1.447,
    ("B", 1.0): 2.017,
    ("B", 0.0): -1.811,
}

PAPER_EXAMPLE_GLOBAL_ONLY = {
    ("A", 1.5): 1.044,
    ("A", 1.0): 0.285,
    # The paper prints 1.732 here; see the erratum note in the module docstring.  The
    # global term cannot depend on the subgroup, and 0.285 + 1.732 = 2.017 matches the
    # paper's own EIS value, so 0.285 is the arithmetically correct figure.
    ("B", 1.0): 0.285,
    ("B", 0.0): -1.234,
}

PAPER_EXAMPLE_SEPARATE = {
    # Plain per-state GRPO, i.e. what the local term alone produces.
    ("A", 1.5): 0.577,
    ("A", 1.0): -1.732,
    ("B", 1.0): 1.732,
    ("B", 0.0): -0.577,
}


def paper_example(component: str = "eis") -> Dict[Tuple[str, float], float]:
    """Expected values for the paper's Appendix D.1 example."""
    if component not in ("eis", "global_only", "separate"):
        raise ValueError("component must be eis/global_only/separate")
    return {
        "eis": PAPER_EXAMPLE,
        "global_only": PAPER_EXAMPLE_GLOBAL_ONLY,
        "separate": PAPER_EXAMPLE_SEPARATE,
    }[component]


def compute_paper_example(component: str = "eis") -> Dict[Tuple[str, float], float]:
    """Recompute the paper's example with the shipped code, grouped for comparison."""
    result = compute_eis_advantages(_paper_example_rollouts())
    values: Dict[Tuple[str, float], float] = {}
    for item in result["components"]:
        if component == "eis":
            value = item["global_term"] + item["local_term"]
        elif component == "global_only":
            value = item["global_term"]
        elif component == "separate":
            value = item["local_term"]
        else:  # pragma: no cover - guarded by paper_example()
            raise ValueError(component)
        key = (item["state_id"], item["reward"])
        if key in values and abs(values[key] - value) > PAPER_TOLERANCE:
            raise AssertionError(
                "paper example is not reproducible: {0} has two different values".format(key))
        values[key] = value
    return values


__all__ = [
    "EPSILON",
    "EISError",
    "PAPER_EXAMPLE",
    "PAPER_EXAMPLE_GLOBAL_ONLY",
    "PAPER_EXAMPLE_SEPARATE",
    "PAPER_TOLERANCE",
    "Rollout",
    "advantages_by_key",
    "compute_eis_advantages",
    "compute_paper_example",
    "group_rollouts",
    "paper_example",
]
