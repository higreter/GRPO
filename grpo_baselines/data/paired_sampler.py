"""Deterministic paired sampling with complete original/swapped states."""
from __future__ import annotations
import random
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

VIEW_ORDER: Tuple[str, ...] = ("original", "swapped")
SAMPLE_ID_SEPARATOR = "::"

class SamplerError(ValueError):
    pass

def split_sample_id(sample_id: Any) -> Tuple[str, str]:
    pair_id, separator, view = str(sample_id).rpartition(SAMPLE_ID_SEPARATOR)
    if not separator or not pair_id or not view:
        raise SamplerError("sample_id must be '<pair_id>::<view>': {0!r}".format(sample_id))
    return pair_id, view

def build_pair_layout(sample_ids: Sequence[Any], expected_states: Optional[int] = None
                      ) -> List[Tuple[str, Dict[str, int]]]:
    grouped: Dict[str, Dict[str, int]] = {}
    for index, sample_id in enumerate(sample_ids):
        pair_id, view = split_sample_id(sample_id)
        if view not in VIEW_ORDER:
            raise SamplerError("unsupported view: {0}".format(view))
        views = grouped.setdefault(pair_id, {})
        if view in views:
            raise SamplerError("duplicate state {0} for pair {1}".format(view, pair_id))
        views[view] = index
    expected_states = expected_states or len(VIEW_ORDER)
    layout = []
    for pair_id, views in grouped.items():
        if len(views) != expected_states:
            raise SamplerError("pair {0} has {1} states, expected {2}".format(
                pair_id, len(views), expected_states))
        expected = set(VIEW_ORDER[:expected_states])
        if set(views) != expected:
            raise SamplerError("pair {0} has incomplete states".format(pair_id))
        layout.append((pair_id, views))
    if not layout:
        raise SamplerError("no samples")
    return layout

class PairedStateSampler:
    """One generation batch contains both full states for every sampled pair."""
    def __init__(self, layout: Sequence[Tuple[str, Dict[str, int]]], *,
                 pairs_per_generation: int, rollouts_per_state: int,
                 num_generations: int, steps: int, seed: int = 0,
                 shuffle: bool = True) -> None:
        if pairs_per_generation < 1 or steps < 1:
            raise SamplerError("pairs_per_generation and steps must be positive")
        if rollouts_per_state < 2:
            raise SamplerError("GRPO needs at least two rollouts per state")
        states = len(next(iter(layout))[1])
        if num_generations != states * rollouts_per_state:
            raise SamplerError("num_generations must equal states * rollouts_per_state")
        self.layout = list(layout)
        self.states_per_pair = states
        self.view_order = tuple(next(iter(layout))[1])
        self.pairs_per_generation = pairs_per_generation
        self.rollouts_per_state = rollouts_per_state
        self.num_generations = num_generations
        self.steps = steps
        self.seed = seed
        self.shuffle = shuffle
        self.declared_pairs_per_generation = pairs_per_generation
        self.batch_narrowed = False
        if pairs_per_generation > len(layout):
            raise SamplerError("pairs_per_generation exceeds dataset size")

    def narrow_to_layout(self, declared_pairs_per_generation: int):
        if declared_pairs_per_generation < self.pairs_per_generation:
            raise SamplerError("cannot widen a narrowed sampler")
        self.declared_pairs_per_generation = declared_pairs_per_generation
        self.batch_narrowed = self.pairs_per_generation < declared_pairs_per_generation
        return self

    @property
    def generation_batch_size(self) -> int:
        return self.pairs_per_generation * self.num_generations

    @property
    def rows_per_generation(self) -> int:
        return self.pairs_per_generation * self.states_per_pair

    def __len__(self) -> int:
        return self.steps * self.generation_batch_size

    def pair_schedule(self) -> List[List[int]]:
        rng = random.Random(self.seed)
        schedule = []
        order: List[int] = []
        cursor = 0
        for _ in range(self.steps):
            batch = []
            while len(batch) < self.pairs_per_generation:
                if cursor >= len(order):
                    order = list(range(len(self.layout)))
                    if self.shuffle:
                        rng.shuffle(order)
                    cursor = 0
                batch.append(order[cursor]); cursor += 1
            schedule.append(batch)
        return schedule

    def batch_layout(self) -> List[List[Tuple[str, str, int, int]]]:
        outcome = []
        for batch in self.pair_schedule():
            rows = []
            for pair_index in batch:
                pair_id, views = self.layout[pair_index]
                for view in self.view_order:
                    for rollout_index in range(self.rollouts_per_state):
                        rows.append((pair_id, view, views[view], rollout_index))
            outcome.append(rows)
        return outcome

    def __iter__(self) -> Iterator[int]:
        for batch in self.batch_layout():
            for _pair_id, _view, row_index, _rollout_index in batch:
                yield row_index

    def __repr__(self) -> str:
        return f"PairedStateSampler(pairs={len(self.layout)})"

def build_sampler(sample_ids: Sequence[Any], *, pairs_per_generation: int,
                  num_generations: int, steps: int, seed: int = 0,
                  shuffle: bool = True) -> PairedStateSampler:
    layout = build_pair_layout(sample_ids)
    states = len(layout[0][1])
    if num_generations % states:
        raise SamplerError("num_generations must be divisible by states")
    return PairedStateSampler(layout, pairs_per_generation=pairs_per_generation,
        rollouts_per_state=num_generations // states, num_generations=num_generations,
        steps=steps, seed=seed, shuffle=shuffle)

def describe_sampler(sampler: PairedStateSampler, batches: int = 1) -> Dict[str, Any]:
    layouts = sampler.batch_layout()
    return {"pairs": len(sampler.layout), "states_per_pair": sampler.states_per_pair,
        "pairs_per_generation": sampler.pairs_per_generation,
        "rollouts_per_state": sampler.rollouts_per_state,
        "num_generations": sampler.num_generations, "steps": sampler.steps,
        "seed": sampler.seed, "generation_batch_size": sampler.generation_batch_size,
        "rows_per_generation": sampler.rows_per_generation, "batches": layouts[:batches],
        "batch_narrowed": sampler.batch_narrowed,
        "declared_pairs_per_generation": sampler.declared_pairs_per_generation}
