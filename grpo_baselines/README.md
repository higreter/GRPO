# GRPO baselines for LLM-as-a-Judge

Four GRPO variants trained on the same base model, the same preference data and the
same reward-bearing surface, so that the only thing that differs between runs is the
algorithm under test:

| Method | Reward | Advantage | States per sample |
|---|---|---|---|
| **Vanilla** | verdict correctness | GRPO group normalisation (TRL `scale_rewards=group`) | 1 |
| **J1** | correctness + *verdict consistency* (both orderings correct) | GRPO group normalisation, groups are per state | 2 (original + swapped) |
| **J4R** | judgement correctness + format bonus/penalty | **EIS**: global term across both states plus local term within the state | 2 (original + swapped) |
| **JwC** | RL-Brier `1 − (p − y)²` on a verbalised probability | GRPO group normalisation | 1 |

Status of this tree, stated plainly: **the CPU tests pass, and nothing else was run.**
No GPU smoke test, no training step, no RewardBench inference. There is no checkpoint
in `runs/` and no prediction file anywhere. See
`../docs/IMPLEMENTATION.md` for the implementation details.

## What is here

```
common/        method-agnostic primitives: labels, prompts, parsers, config, paths
data/          adapters (HelpSteer3, RewardBench), paired sampler, length audit
rewards/       reward functions: correctness, J1 pair reward, JwC Brier
trainers/      TRL wiring; one package per method plus the J4R EIS machinery
evaluation/    RewardBench runner (two orderings) and the metric definitions
configs/       four method configs plus the controlled-budget table
scripts/       CPU tests, length audit, optional Smoke, and multi-GPU launch scripts
requirements/  pinned instructions for the two environments this code expects
tests/         standard-library unittest suite (428 tests)
```

### The label space

Two different alphabets appear everywhere and must never be confused:

* `answer1` / `answer2` — the **physical** responses from the source dataset.
* `A` / `B` — the **display** slots the judge sees.

The swapped view puts `answer2` in slot A, so the correct letter flips while the
correct *answer* does not. Every conversion goes through `common/labels.py`; nothing
else is allowed to hardcode the mapping. `J1`'s consistency term and the evaluation's
position-consistency metric both compare physical identities, never display letters —
a judge that always answers "A" is perfectly letter-consistent and worthless.

### The two prompt-shape rules

`data/trl_dataset.py` emits `prompt` as a **messages list**, never a rendered string.
TRL's `is_conversational` only recognises `[{"role", "content"}]`; a plain string
silently takes the non-conversational tokenisation path and the chat template is
skipped. That failure mode is invisible in the loss and would quietly train on a
different prompt than the one that was audited.

`configs/controlled_budget.yaml` is the record of *why* the four runs are comparable.
It holds the shared configuration, the per-method geometry, a paper-vs-controlled
table and the open risks. `common/config.py` refuses to load a config whose derived
numbers do not multiply out, because a budget table that can drift is not a budget.

## The per-rollout log

Every rollout keeps `prompt_id`, `pair_id`, `view`, `rollout_index`, the raw completion,
the reward, the parser's reason code, the parsed outcome, the format flag and its group
key (`rewards/rollout_log.py`). After each generation batch the trainer appends four
numbers to `state.log_history`:

| Metric | Meaning |
|---|---|
| `rollout_log/flat_ratio` | share of rollouts in groups whose rewards are all equal — those groups get advantage exactly 0 and teach nothing |
| `rollout_log/underfilled` | groups with fewer than two *valid* rollouts |
| `rollout_log/invalid` | rollouts the parser could not read |
| `rollout_log/groups` | groups in the batch |

Flat and underfilled are different questions and the code keeps them apart. A flat group
is judged on the rewards TRL actually normalises over, invalid `0.0`s included. An
underfilled group is judged on the reason code, because an unparsable completion still
carries a reward (`0.0`, or `-0.5` for J4R's format penalty) — "has a reward" is not the
same as "is a signal".

## Running the CPU tests

```bash
bash scripts/test_cpu.sh              # 428 tests, no GPU required
bash scripts/test_cpu.sh tests.test_j4r_advantages -v
```

The suite is `unittest`, not `pytest`: the sanctioned environment
(`/data1/yanghao/miniconda3/envs/judgelrm-sft`) has Python 3.9 and no pytest, and
nothing may be installed. Most of the suite imports no third-party package at all;
the two files that read the real datasets only need the standard library.

What the tests *cannot* prove: that any of this converges. They check label algebra,
group geometry, the EIS arithmetic against the paper's worked example, reward
arithmetic, and the evaluation metric definitions. Convergence needs a real GPU run; CPU tests alone cannot establish it.

## What has not been run

```bash
# never executed -- prints the plan only, works without TRL
python -m trainers.j4r.trl_trainer --config configs/j4r.yaml --dry-run

# never executed -- requires the trl-py311 environment and at least one CUDA GPU.
# Refuses with exit 4 unless you confirm: it would load the 3B weights and train.
GRPO_CONFIRM_TRAIN=yes bash scripts/train_j4r.sh --skip-smoke --skip-smoke

# never executed -- requires a checkpoint that does not exist
bash scripts/eval_rewardbench.sh j4r runs/j4r/final
```

`--dry-run` is the one entry point that does work here: it prints the resolved plan
(dataset census, sampler layout, TRL kwargs, budget, advantage identity) as JSON and
touches no model. When `--limit` narrows the batch below `global_batch_size`, the plan
says so in `debug.budget_comparable` rather than presenting a debug run as the
controlled budget.

## Environment

Training uses one of two documented environment sheets:

* `requirements/trl-py311.txt` — TRL 1.18.0.dev0, Python ≥ 3.11.
* `requirements/verl-py310.txt` — verl, Python ≥ 3.10,<3.13 (the J4R estimator also
  ships as a verl `adv_estimator`, but the TRL path is the primary one).

`requirements/README.md` lists dependency version compatibility requirements.

## Provenance

`common/paths.py:UPSTREAM_PINS` records the exact commit of every upstream tree this
code was written against. `external/` and `database/` are read-only inputs; this
project writes only under its own root and `common/paths.py` enforces that.
