# GRPO Baselines — Implementation Notes

Implementation root: `/data1/yanghao/paper/GRPO/grpo_baselines/`
Report date: 2026-11-04
Model: `/data/common_data/Qwen/Qwen2.5-3B-Instruct` (fixed, 32 768-token context)
Training source: `/data1/yanghao/paper/GRPO/database/helpsteer3_train_clean.jsonl` (22 190 rows)
OOD evaluation source: `/data1/yanghao/paper/GRPO/database/rewardbench_v1_ood_clean.jsonl` (2 985 rows)

## 0. Status, stated before anything else

| | Status |
|---|---|
| Code written | 63 Python files, 12,742 lines; 5 configs; 12 scripts; 3 requirement sheets |
| CPU tests | **Passed** — `Ran 428 tests / OK` |
| GPU smoke test | **Not run** |
| Full training | **Not run** |
| RewardBench inference | **Not run** |

Nothing in this report describes an observed training outcome. Every number is either
measured on CPU (data census, token lengths, unit-test arithmetic) or copied from a
paper. There is no checkpoint, no prediction file and no reward curve.

Two hard constraints shaped the whole job: no installation into
`/data1/yanghao/miniconda3/envs/judgelrm-sft` (which has Python 3.9 and no TRL, verl,
vLLM, ray, datasets or pytest), and no training or inference of any kind. The
consequence is that the four trainers are written against TRL's extension points and
verified structurally, not numerically.

---

## 1. The four methods, their papers and their entry points

| Method | Paper | Reward | Advantage | Entry point |
|---|---|---|---|---|
| Vanilla | — (GRPO baseline) | `1` if verdict correct else `0` | GRPO group normalisation | `trainers/vanilla/trl_trainer.py` |
| J1 | *J1: Incentivizing Thinking in LLM-as-a-Judge via Reinforcement Learning* (arXiv:2505.10320v3), Sec. 2.2 | correctness `+1`; consistency `+1` only when **both orderings** are correct | GRPO group normalisation, groups per state | `trainers/j1/trl_trainer.py` |
| J4R | *J4R: Learning to Judge with Equivalent Initial State Group Relative Policy Optimization* (arXiv:2505.13346v3), Sec. 3.2 Eq. 4, Sec. 4 Eq. 6 | judgement `1/0` **plus** format `±0.5` | **EIS**: global + local normalisation | `trainers/j4r/trl_trainer.py` |
| JwC | *Judging with Confidence: Calibrating Autoraters to Preference Distributions* (arXiv:2510.00263), Sec. 3 Eq. 1 | RL-Brier `1 − (p − y)²`, unparsable → `0` | GRPO group normalisation | `trainers/jwc/trl_trainer.py` |

### 1.0 Provenance, one row per method

| Method | Paper, section, equation | Upstream code actually used | Git URL and commit |
|---|---|---|---|
| Vanilla GRPO | No baseline paper of its own. Objective from *DeepSeekMath* (arXiv:2402.03300), Sec. 4.1 — the group-relative advantage `Â = (R − mean(R)) / std(R)` and the clipped surrogate | TRL `GRPOTrainer` loss, advantage and KL (`external/trl`). **No code from the DeepSeek-Math pin was used** — that repository is the paper's own release (models and evaluation materials) and contains no GRPO trainer | https://github.com/huggingface/trl.git @ `3885117d6a1dd647d43befeddf0da482c505cb47` (code); https://github.com/deepseek-ai/DeepSeek-Math.git @ `b8b0f8ce093d80bf8e9a641e44142f06d092c305` (paper's release, recorded in `common/paths.py:UPSTREAM_PINS`, not imported) |
| J1 | *J1: Incentivizing Thinking in LLM-as-a-Judge via Reinforcement Learning* (arXiv:2505.10320v3), Sec. 2.2 (the two rewards), Sec. 2.3 (PaV), Appendix C (hyperparameters), Fig. 9 (prompt) | TRL trainer + reward protocol; no official J1 training code is public | https://github.com/huggingface/trl.git @ `3885117d6a1dd647d43befeddf0da482c505cb47` |
| J4R | *J4R: Learning to Judge with Equivalent Initial State Group Relative Policy Optimization* (arXiv:2505.13346v3), Sec. 3.2 Eq. 4 (EIS advantage), Sec. 4 Eq. 6 (dual reward), Appendix D.1 (worked example, with the erratum noted below) | Evaluation code only — prompt template and verdict parsing rule, from `reasoning_judge_inf/prompts/default_prompts.py` | https://github.com/SalesforceAIResearch/ReasoningJudgeBench.git @ `e5a6c64fc97dd65f2b1b868664fb12c2d12b0606`; also https://github.com/volcengine/verl.git @ `745432653d15af14954af1e055c8aeed72afa064` for the optional `j4r_eis` estimator |
| Judging with Confidence (JwC) | *Judging with Confidence: Calibrating Autoraters to Preference Distributions* (arXiv:2510.00263), Sec. 3 Eq. 1 (Brier reward), Eq. 2 (log reward), Prop. 1, Appendix C.1 (ECE/Brier definitions), Table 5 (RL settings) | TRL `GRPOTrainer` loss, advantage and KL (`external/trl`); the Brier reward is implemented here, since the paper ships no training code | https://github.com/huggingface/trl.git @ `3885117d6a1dd647d43befeddf0da482c505cb47` |

All four pins are also recorded in `common/paths.py:UPSTREAM_PINS`, and
`tests/test_pins.py` compares them against the trees on disk: each pin's commit must
equal that repository's `git rev-parse HEAD`, `git status --porcelain` must be empty and
`git diff <pin> HEAD` must be empty. That is what makes "we did not modify them" a
checkable statement rather than a claim. Two of the four are provenance only: the DeepSeek-Math release
supplies the paper, not a trainer, and is never imported.

The verl pin appears on the J4R row rather than a row of its own: `trainers/j4r/verl_adv.py`
is the only module in the project that touches verl, and it is an optional alternative to
the TRL injection path. Vanilla, J1 and JwC run on TRL alone.

No official training code exists for J1, J4R or JwC. J4R's organisation publishes
evaluation code only; the prompt template and the `Verdict:` parsing rule were taken from
it, and everything else — reward composition, EIS advantage, trainer wiring — was
re-derived from the papers. This is an independent implementation of the published
objectives, not a reproduction of a reference run.

Shared plumbing lives in `trainers/grpo_base.py` (dataset → sampler → TRL kwargs →
trainer) and `trainers/trl_bridge.py` (config translation, dependency guards,
manifest). A method package supplies only what is genuinely method-specific.

### 1.1 J1's consistency term is not "the two letters match"

In the swapped view `answer2` sits in slot A, so a judge that correctly prefers the
same physical response emits **different letters** in the two views. The reward
therefore compares response identity, not letters:

```python
# rewards/j1.py
both_correct = original_correct and swapped_correct   # the only definition of consistency
```

`tests/test_j1.py::PositionBiasTest` pins both directions: a position-neutral judge
gets `same_response_chosen=True` while its two letters differ, and a judge that always
answers "A" gets `same_response_chosen=False`. There is no path through the code where
letter equality can be substituted for identity.

### 1.2 J4R's EIS is an advantage rewrite, not a reward name

Paper equation (Sec. 3.2 Eq. 4):

```
Â^(i,ℓ) = (R^(i,ℓ) − R̄_[L]) / σ_R_[L]   +   (R^(i,ℓ) − R̄_ℓ) / σ_R_ℓ
              global term                        local term
```

Implemented in `trainers/j4r/eis.py:compute_eis_advantages`, with population standard
deviation (divide by N). Population σ is not a stylistic choice: it is the only
variance that reproduces the paper's own worked example in Appendix D.1. verl's
`group_mean_std` and TRL's `nanstd` both use sample variance, so the code deliberately
diverges from both and says so in a docstring.

**A paper erratum, found by the worked example.** Appendix D.1's "Global advantage
only" paragraph prints `+1.732` for subgroup B's reward-1.0 rollouts. That value is
subgroup B's *local* term. The global term cannot depend on the subgroup: the global
mean is `26/32 = 0.8125` and the global population σ is `0.6585`, so
`(1.0 − 0.8125) / 0.6585 = 0.285`. The same paragraph's EIS value `2.017 = 0.285 +
1.732` confirms it. `PAPER_EXAMPLE_GLOBAL_ONLY[("B", 1.0)]` therefore stores `0.285`,
and `tests/test_j4r_advantages.py::PaperExampleTest` asserts against `0.285` within
`5e-4`. Both the erratum and the tolerance are documented at the constant.

Verified numerically:

| component | A/1.5 | A/1.0 | B/1.0 | B/0.0 |
|---|---|---|---|---|
| EIS (global + local) | 1.621 | −1.447 | 2.017 | −1.811 |
| global only | 1.044 | 0.285 | 0.285 | −1.234 |
| separate groups (vanilla) | 0.577 | −1.732 | 1.732 | −0.577 |

The two-term form is required, not decorative: the paper reports that using the global
term alone drops JudgeBench from 56.86 to 49.14, and the local term alone is just
ordinary GRPO with a different grouping. `tests/test_j4r_advantages.py::
NotVanillaNormalisationTest` asserts that EIS and prompt-local normalisation produce
different numbers on the same reward vector.

The estimator is also registered for verl
(`trainers/j4r/verl_adv.py:ADV_ESTIMATOR_NAME = "j4r_eis"`), because verl's
`compute_advantage` hands a custom estimator only the `uid` column — so the two-level
`pair → state` identity has to travel inside the uid string as
`"<pair_id>::<view>"`. `tests/test_verl_adv.py` exercises that path against a stand-in
`core_algos`, so verl does not have to be installed to check it: the same hand-worked
batch is reproduced through the tensor plumbing, plain `"0"`/`"1"` uids raise rather
than degrading to per-group GRPO, and a state with a single rollout is rejected because
its local term has no σ.

One detail the tests had to pin down, because it is easy to read the output wrongly:
`einsum_advantage` returns `(bs, response_length)` with the sequence advantage
*broadcast* over every unmasked token. Reading a row's advantage means taking one token
column, not summing the row — the sum is `response_length` times too large. And a
zero-variance state is not a zero advantage: its local term is `0` while its global term
is whatever the pair-wide normalisation says (`-0.7071` in the case
`tests/test_verl_adv.py` uses). Only a fully flat *pair* yields all zeros.

### 1.3 JwC's probability head

`rewards/jwc_brier.py` implements `R = 1 − (p − y)²` with `y = 1` meaning *B better*
and `p = Pr[B ≻ A]`. Unparsable output scores `0`, which is the paper's unparsable
branch, not a neutral `0.25`. Brier reward is **not** clipped (it is a quadratic in
`[0,1]`); only the log reward clips, at `ε = 1e-3`.

The paper's main experiments use a *dense* preference distribution (ten labels per
prompt). HelpSteer3 gives one hard label per pair, so this implementation is a
**sparse-label adaptation** of RL-Brier and says so at the top of the module, in the
config comment, in the launch script and in the controlled-budget table. It is a
weaker objective and the report treats it as such.

---

## 2. Data handling and isolation

### 2.1 Verified census (read from the real files)

| File | Rows | Labels | Views |
|---|---|---|---|
| `/data1/yanghao/paper/GRPO/database/helpsteer3_train_clean.jsonl` | 22 190 | A 11 068 / B 11 122 | all `original` |
| `/data1/yanghao/paper/GRPO/database/rewardbench_v1_ood_clean.jsonl` | 2 985 | A 1 495 / B 1 490 | 23 official subsets |

Both paths are constants in `common/paths.py` (`HELPSTEER_TRAIN`, `REWARDBENCH_OOD`) and
`common/config.py` compares them by realpath, so a config cannot silently point at a copy
elsewhere on disk. A third file, `helpsteer3_test_clean.jsonl` (1 866 rows), is present and
deliberately unused: the brief allows it as a structural reference but forbids tuning the
RewardBench numbers against an in-domain hold-out, and nothing here reads it.

`label == "A"` means `answer1_body` is better; `label == "B"` means `answer2_body` is
better. This was confirmed by crossing the label against
`source_overall_preference`, not assumed from field order.

### 2.2 Field contract

The intermediate row carries exactly the fields the brief requires
(`sample_id, pair_id, question, answer_a, answer_b, ground_truth, is_swapped,
source_dataset`) plus the audit columns `view`, `better_response`,
`source_row_index`, `answer_1`, `answer_2`. `data/helpsteer_adapter.py` lists
`FORBIDDEN_OUTPUT_FIELDS = ("text", "source_overall_preference")` and raises if either
leaks into a training row.

That `text` field matters: `database/README.md` documents it as the *old JudgeLRM
rationale* (`"<think>@Response 1 is much better…"`). It is not the template this
project trains on, and importing it would silently train a different task.
`tests/test_adapters.py::HelpSteerAdapterTest` serialises a full prompt and asserts the
legacy phrase never appears.

**Traceability is carried, not reconstructible.** `source_line` and `group_id` travel as
ordinary record fields through four links: source row → `PairView.to_record()` →
`build_trl_row()` → `rewards/rollout_log.py:RolloutRecord`. This chain was broken twice
in the middle and each break was invisible from the ends:

* `PairView.to_record()` emitted eleven keys and dropped `group_id`; only
  `PairView.as_dict()` (which includes `metadata`) had it, so the data looked preserved.
* `build_trl_row()` rebuilt its row from a whitelist and dropped the field again, so even
  after the first fix it never reached TRL, the reward function, or the log.

Both are repaired and `tests/test_traceability.py` walks the whole chain on real rows,
including a check that the `group_id` a row reports is the one in
`helpsteer3_train_clean.jsonl` for that `source_line` — not merely non-empty. The practical
payoff is in Sec. 5.2: a flat group can be named, not just counted. `pair_id` remains
`hs3-<source_line>`, so recovering one recovers the other, but neither is now the only
route back to the source.

### 2.3 RewardBench never touches training

Three independent guards:

1. `common/config.py` rejects any `train_dataset` whose realpath is not
   `helpsteer3_train_clean.jsonl`, with the message *"RewardBench must never become a
   training file."*
2. `data/rewardbench_adapter.assert_not_training(path)` refuses the OOD file.
3. `tests/test_adapters.py::RewardBenchIsolationTest` asserts no RewardBench row can
   be turned into a training row.

### 2.4 A real defect found and fixed during testing

`data/rewardbench_adapter.py` originally declared 23 subsets that did not match the
file: it listed `chat-hard`, `chat-medium`, `chat-easy` (which do not exist) and
omitted `xstest-should-refuse` (154 rows) and `xstest-should-respond` (250 rows).
Unfixed, the per-subset table would have grown three phantom subsets and dropped 404
real rows from the official aggregate — a plausible-looking but wrong number. Fixed by
rewriting `OFFICIAL_SUBSETS` from the measured file and adding a hard check in
`convert_row`, so a mis-spelled subset now fails at load time instead of quietly
shrinking a denominator.

---

## 3. Length audit — real tokenizer, real templates, every row

Measured with `Qwen2TokenizerFast` and each method's own chat template rendered through
`apply_chat_template(..., add_generation_prompt=True)`; the count is the length of that
exact string. `scripts/audit_lengths.py` reproduces this.

| Method | Rows | min | p50 | p90 | p95 | p99 | max |
|---|---|---|---|---|---|---|---|
| Vanilla | 22 190 | 191 | 1 330 | 3 192 | 3 783 | 4 809 | **10 516** |
| JwC | 22 190 | 265 | 1 404 | 3 267 | 3 857 | 4 883 | **10 590** |
| J1 | 44 380 | 298 | 1 437 | 3 300 | 3 890 | 4 916 | **10 623** |
| J4R | 44 380 | 286 | 1 425 | 3 287 | 3 878 | 4 904 | **10 611** |

Longest sample for all four: `hs3-9274::original`. σ = 1 067.15 in every row because
all four methods render the same 22 190 pairs and the templates differ only in a fixed
prefix.

With `max_prompt_length = 11 264` and `max_completion_length = 1 024`:

* **Zero prompts exceed the limit** in any method (10 623 < 11 264).
* Total budget `11 264 + 1 024 = 12 288 ≤ 32 768`, leaving 20 480 tokens of headroom.
* `assert_within_limits` raises `LengthAuditError` rather than truncating or dropping,
  and names the count, ratio, longest sample and overshoot. The audit refuses to be
  quietly satisfied.

The single-state methods audit 22 190 rows and the two-state methods 44 380, because
pairs are duplicated into original + swapped. That doubling is honest: J1 and J4R
really do render two prompts per source pair.

---

## 4. Controlled comparison budget

All four configs share base model, dataset, seed, prompt/completion limits, learning
rate (`1e-6`, constant, no warmup), temperature `1.0`, `top_p 1.0`, clip `ε = 0.2`,
`kl_beta = 0.01` with a frozen initial reference, `global_batch_size = 8`,
`micro_batch_size = 2`. The comparison is matched on **rollouts and generated tokens**,
not on optimizer steps, because a paired-states method spends two prompts per sample
and would otherwise receive twice the gradient signal for the same step count.

| | Vanilla | J1 | J4R | JwC |
|---|---|---|---|---|
| states per sample | 1 | 2 | 2 | 1 |
| `num_generations` (total rollouts / sample) | 8 | 16 | 16 | 8 |
| rollouts per state | 8 | 8 | 8 | 8 |
| dataset rows per update | 8 | 16 | 16 | 8 |
| rollouts per update | 64 | 128 | 128 | 64 |
| gradient accumulation | 4 | 8 | 8 | 4 |
| optimizer steps | 256 | 128 | 128 | 256 |
| **total rollouts** | **16 384** | **16 384** | **16 384** | **16 384** |
| **estimated generated tokens** | **46 268 416** | **46 268 416** | **46 268 416** | **46 268 416** |
| measured generated tokens | *none — nothing was trained* | *none* | *none* | *none* |
| frozen reference + KL | yes, `kl_beta = 0.01` | yes, `0.01` | yes, `0.01` | yes, `0.01` |

`configs/controlled_budget.yaml` records the paper-vs-controlled table (ten axes) and
five open risks. Each method row also carries its own `estimated_generated_tokens`,
`actual_generated_tokens` and `frozen_reference_with_kl`, so all eight facts Sec. 9 asks for
are answerable one row at a time; `actual_generated_tokens` is `null` in every row and says
why. `tests/test_budget_table.py` pins that the rows agree with the configs and that a
method cannot silently drop its KL side or claim a measured token count. `common/config.py` recomputes every derived quantity and refuses a
config whose numbers do not multiply out, so this table cannot drift away from the
YAML files.

This is not an ad-hoc choice. The J4R paper runs the same ablation when it compares its
duplicated-ordering baseline against EIS-GRPO: it uses "G = 16 and doubles the rollout
batch size, resulting in the same number of model updates and training steps"
(*J4R*, Sec. 5 discussion of [Dup], `papers/j4r_2505.13346.txt:1285`). Holding the work
fixed and letting the step count fall out is the paper's own way of keeping a paired-state
method comparable.

An update holds 64 rollouts for the single-state methods and 128 for the paired ones
(`rows_per_update × rollouts_per_state`, which `common/config.py` re-derives and
`tests/test_configs.py` pins), and each sequence is bounded by
`max_prompt_length + max_completion_length = 12 288` tokens against the model's 32 768.
The `rollouts_per_state ≥ 2` floor is enforced because a GRPO group of size one has no
variance to normalise against.

### 4.1 Paper-original versus controlled

| Axis | Paper | Here | Why |
|---|---|---|---|
| Base model | Llama-3.1-8B / 70B, Qwen2.5-7B, Gemma-2-9B | Qwen2.5-3B-Instruct | fixed by the brief; only *relative* method differences are interpretable at 3B |
| Rollouts per prompt | J1 5, J4R 32, JwC 32 | 8 per state | CPU-verified plumbing, single GPU, 16 384 total rollouts |
| KL coefficient | J1 0.01→0, J4R 1e-4, JwC 0.01 | 0.01 everywhere | a differing KL would be a method effect, not a method |
| Learning rate | J1 1e-6, JwC 3e-7 | 1e-6 everywhere | same reason |
| Sequence budget | J1 4 096, JwC 2 048 | 11 264 + 1 024 | the measured HelpSteer distribution needs it; no truncation |
| Reward (J4R) | judgement + format | same | paper Eq. 6 |
| Reward (JwC) | dense preference distribution | sparse hard label | HelpSteer has one label per pair; **weaker objective** |
| Advantage | GRPO / EIS as published | same | the thing under test |

---

## 5. Upstream code reused, and what had to be extended

| Repository | Commit | Version / use |
|---|---|---|
| https://github.com/huggingface/trl.git | `3885117d6a1dd647d43befeddf0da482c505cb47` | 1.18.0.dev0 — trainer, config, reward protocol |
| https://github.com/volcengine/verl.git | `745432653d15af14954af1e055c8aeed72afa064` | advantage-estimator registry for the EIS registration |
| https://github.com/SalesforceAIResearch/ReasoningJudgeBench.git | `e5a6c64fc97dd65f2b1b868664fb12c2d12b0606` | J4R prompt template and verdict-parsing rule only (no training code) |
| https://github.com/deepseek-ai/DeepSeek-Math.git | `b8b0f8ce093d80bf8e9a641e44142f06d092c305` | GRPO reference for the baseline loss |

Nothing under `external/` was modified — `git status --porcelain` is empty in all four
trees — and `common/paths.py:UPSTREAM_PINS` records the same commits so the pins travel
with the code.

### 5.1 Used as-is

* `GRPOTrainer` + `GRPOConfig` for the PPO-clipped surrogate, the reference-model KL
  (via `beta`), generation, and the `steps_per_generation` gradient-accumulation split.
* Reward function protocol: `reward_func(prompts=, completions=, completion_ids=,
  **reward_kwargs)`, one scalar per completion, `None` → NaN.
* Custom dataset columns: `_calculate_rewards` forwards every non-prompt column, so
  `sample_id`, `pair_id`, `ground_truth`, `view` reach the reward function without a
  custom collator (TRL installs `data_collator=identity`).

### 5.2 Extended, and why

**The paired sampler (all four methods).** TRL's `RepeatSampler` repeats each prompt
`num_generations` times and shuffles. J1 and J4R need the original and swapped views of
one pair inside the *same* generation batch, adjacent and in fixed order, or the
consistency term has no counterpart and EIS has no second state.
`data/paired_sampler.py` builds that layout and `trainers/grpo_base.py:SamplerMixin`
returns it from `_get_train_sampler`, bypassing `RepeatSampler`. Falling back is made
impossible: the mixin raises `RuntimeError("no paired sampler was attached…")` rather
than silently degrading.

**Batch-size semantics.** TRL counts `per_device_train_batch_size` in *prompts*; this
project's `micro_batch_size` counts *states*, and one state is one dataset row. The
bridge therefore multiplies by `rollouts_per_state` and cross-checks
`per_device_rows × gradient_accumulation == generation_batch_size`, raising instead of
letting the sampler and dataloader disagree about how many rollouts an update holds.
TRL's `num_generations` is set to `rollouts_per_state`, because TRL groups rewards with
`rewards.view(-1, num_generations)` and a group must be one state.

**J1's batch guard.** `trainers/j1/trl_trainer.py:J1ConsistencyGuardMixin` inspects the
reward function's `last_stats` after each generation batch and raises if any group
lacks its counterpart. Without it, a sampler bug would silently turn J1 into vanilla
and the run would look fine.

**J4R's advantage injection.** `EISAdvantageMixin._generate_and_score_completions`
calls `super()`, then overwrites `output["advantages"]`. It has to read the rewards
from the reward function's `last_rewards` because TRL's `rewards` is a local variable,
and it has to run *after* super() because one generation batch is generated once and
then split into `steps_per_generation` chunks — EIS must normalise over the whole batch,
before the split. Eval mode is handled too: `_prepare_inputs` routes eval through the
same method, so skipping the injection there would give an eval loss computed under a
different advantage rule.

**`--dry-run`.** `run_cli` skips the trainer-class factory when `--dry-run` is set, so
all four entry points can print their plan in an environment without TRL. This was a
real defect found by running it: J1 and J4R died on `MissingDependencyError` before
printing anything.

**The per-rollout log (Sec. 5.4) and the degenerate-group diagnostics (Sec. 5.5).**
TRL is numerically safe here but silent: `advantages = rewards - mean_grouped_rewards`,
then `/ (std_rewards + 1e-4)`, then `torch.nan_to_num(..., nan=0.0)`. A group whose
rewards are all equal therefore yields advantages of exactly zero — no NaN, no warning,
no learning signal. If the parser breaks and every completion in a batch scores `0.0`,
the loss goes flat and the run looks healthy.

`rewards/rollout_log.py` keeps one `RolloutRecord` per rollout (prompt id, pair id,
view, rollout index, raw completion, reward, reason code, outcome, format flag, group
key — the ten fields Sec. 5.4 asks for) and reduces them with `group_diagnostics`,
which separates two things that are easy to conflate:

* *flat* — the rewards TRL actually normalises over are all equal, so the advantage is
  identically zero. Counted over every non-`None` reward, invalid outputs included,
  because those `0.0`s really do enter the mean and σ.
* *underfilled* — fewer than two **valid** rollouts in the group. Validity is decided by
  the reason code, not by "a reward exists": an unparsable completion earns `0.0`
  (vanilla/J1/JwC) or `-0.5` (J4R's format penalty), so it has a reward but is not a
  signal.

Wrapping is done by `logged_reward_fn`, a pass-through that returns the original
values unchanged and attaches `last_rollouts` / `last_diagnostics`; all four entry
points use it, and a test asserts none of them forgot. `RolloutLogMixin` (in
`trainers/grpo_base.py`) then reads the diagnostics after each generation batch and
appends `rollout_log/groups`, `rollout_log/flat_ratio`, `rollout_log/underfilled` and
`rollout_log/invalid` to `state.log_history`. It is composed *before* the method class
in the MRO, so for J4R it observes the batch after EIS has replaced the advantages;
`tests/test_rollout_composition.py` pins that ordering with stand-in bases. A failing
diagnostic prints a warning and lets training continue — losing a counter is not worth
killing a run, but it must not be invisible either.

**Argument validation and resume semantics (Sec. 9).** `validate_args` runs before
anything is loaded: `--limit < 1` is rejected (a limit of zero selects no pairs), and
`--resume-from-checkpoint` must be an existing directory containing `trainer_state.json`
— otherwise the resume point cannot restore the optimizer step or the sampler position
and the failure would surface much later, inside transformers, with an unrelated
traceback.

The shell scripts validate the same things independently, and they do it *before*
activating the environment. That ordering was a bug first: the guard sat after
`conda activate`, and because the target env does not exist on this host, `set -e`
aborted on the activation and the user saw a bare exit 1 with no message. A check you
can only observe in a fully provisioned environment is not a check. Now an unknown flag
or a missing checkpoint exits 2 with a reason, and each script reads `output_dir` out of
its yaml instead of repeating it — otherwise redirecting a run would print the old
destination.

Starting a run also requires a named opt-in, `GRPO_CONFIRM_TRAIN=yes`; without it the
script exits 4 before it sources conda. That gate exists because of an incident: a test
asserted that a launch script stopped with "conda env does not exist", which was true
only until the environment was created. Once it was, the same test started a real
Qwen2.5-3B training run, loaded the weights, and died in CUDA OOM on its first step. A
test that asserts a fact about the machine tests the machine. The gate makes starting
training a decision rather than an accident, and it makes the exit code independent of
what happens to be installed. `tests/test_scripts.py` now asserts four things: unknown
flags and missing checkpoints are rejected with a message, a bare invocation exits 4,
and `runs/` is unchanged across the call.

---

## 6. New files

63 Python files (12,742 lines), 5 YAML, 12 shell/Python scripts, 3 requirement sheets.

Two execution-support additions are kept in-project rather than patched into TRL: the
`dual_device_reference` adapter enforces a single-policy-GPU plus frozen-reference-GPU
layout, and `create_smoke_config.py` writes isolated one-step smoke configs without
editing a formal config or source dataset. They do not alter any method's reward,
advantage, rollout-group size, sequence limit, or controlled formal budget.

```
common/{__init__,config,labels,parsing,paths,prompts,reproducibility}
data/{__init__,helpsteer_adapter,length_audit,pair_views,paired_sampler,
      rewardbench_adapter,trl_dataset}
rewards/{__init__,base,correctness,j1,jwc_brier,rollout_log}
trainers/{__init__,dual_device_reference,grpo_base,trl_bridge}
evaluation/{__init__,metrics,rewardbench}
trainers/vanilla/{__init__,trl_trainer}
trainers/j1/{__init__,trl_trainer}
trainers/j4r/{__init__,advantage,eis,reward,trl_trainer,verl_adv}
trainers/jwc/{__init__,trl_trainer}
configs/{controlled_budget,j1,j4r,jwc_rl_brier,vanilla}
scripts/{audit_lengths.py,build_final_report.py,create_smoke_config.py,
         eval_rewardbench.sh,test_cpu.sh,train_j1.sh,train_j4r.sh,
         train_jwc.sh,train_vanilla.sh,validate_checkpoint.py,
         watch_and_launch_formal.sh,watch_and_launch_smokes.sh}
requirements/{README.md,trl-py311.txt,verl-py310.txt}
tests/{__init__,test_adapters,test_budget_table,test_cli_args,test_config_prose,
       test_configs,test_dry_run,test_evaluation,test_j1,test_j4r_advantages,
       test_jwc_brier,test_parsing_and_rewards,test_pins,test_report_consistency,
       test_rollout_composition,test_rollout_health,test_rollout_log,test_scripts,
       test_traceability,test_validation_purity,test_verl_adv,test_writers}
README.md
```

Generated dry-runs and setup logs are not committed. `runs/` starts empty; Smoke, formal
training and evaluation create their own auditable outputs there. No checkpoint,
prediction file or training log is present — the absence of those is what "Full training:
Not run" means in practice, and
`tests/test_validation_purity.py` plus the audit script both assert that merely loading a
config still creates nothing.

**Declarations.** Nothing under `database/`, `external/`, `papers/`,
`/data/common_data/Qwen/Qwen2.5-3B-Instruct/` or the conda environment was created,
modified or deleted. `common/paths.py` encodes the writable root and raises
`PermissionError` outside it; the test suite asserts that refusal positively
(`test_writers_refuse_paths_outside_the_project_root`). No package was installed.

J4R's reward lives in `trainers/j4r/reward.py` rather than `rewards/` on purpose: its
`±0.5` format term is defined on J4R's `Explanation:` / `Verdict: [[A>B]]` layout and is
inseparable from the EIS grouping that consumes it.

---

## 7. Environment and the installation that was not performed

`/data1/yanghao/miniconda3/envs/judgelrm-sft` — Python 3.9.21, torch 2.8.0+cu128,
transformers 4.57.6, tokenizers 0.22.2, accelerate 1.10.1, deepspeed 0.19.7,
safetensors 0.7.0, numpy 2.0.2, pyyaml 6.0.3, jinja2 3.1.6. **Absent:** trl, verl,
vllm, ray, datasets, peft, flash_attn, wandb, sentencepiece, **pytest** (all confirmed
absent by import, not by reading a lockfile). Hence `unittest`, not `pytest`, and hence a
mostly dependency-free suite.

`requirements/trl-py311.txt` (TRL 1.18.0.dev0, Python ≥ 3.11) and
`requirements/verl-py310.txt` (Python ≥ 3.10,<3.13) are **instruction sheets, not
executed installs**; each says so in its first lines.

The commands they describe, quoted here so that "not executed" is checkable rather than
asserted. **None of these was run.** `judgelrm-sft` cannot even host them (Python 3.9
against TRL's `requires-python = ">=3.11"`), and installing into it was forbidden:

```bash
# --- TRL path (all four methods) -------------------------------------------
conda create -n grpo-trl-py311 python=3.11 -y
conda activate grpo-trl-py311
pip install -r requirements/trl-py311.txt
pip install "trl @ git+https://github.com/huggingface/trl.git@3885117d6a1dd647d43befeddf0da482c505cb47"

# --- verl path (alternative EIS estimator) ---------------------------------
conda create -n grpo-verl-py310 python=3.10 -y
conda activate grpo-verl-py310
pip install -r requirements/verl-py310.txt
pip install "verl @ git+https://github.com/volcengine/verl.git@745432653d15af14954af1e055c8aeed72afa064"
```

Recorded hazards:

1. **vLLM / torch / CUDA ABI.** Use a PyTorch/CUDA wheel compatible with the installed driver; a
   mis-matched pair fails at import or at first kernel launch, not at `pip`.
2. **FlashAttention build order.** It must compile against a frozen torch; upgrading
   torch afterwards invalidates the extension.
3. **NCCL / `ray.util.collective`.** verl's multi-GPU path additionally wants
   `cupy-cuda13x`; the CUDA major version must match the installed toolkit.
4. **torch pin drift.** verl-core pins `torch==2.9.1`, its `vllm` extra pins
   `torch==2.13.0`, the read-only environment has 2.8.0. These three cannot coexist;
   each environment needs its own.
5. **TRL API drift.** The J4R injection depends on
   `_generate_and_score_completions` returning an `advantages` key and on
   `_calculate_rewards` forwarding custom columns. Both are true at the pinned commit
   and are asserted by reading the pinned source in
   `tests/test_parsing_and_rewards.py::GroupAdvantageContractTest`, which will fail
   loudly if the pin moves.

**No algorithm logic was deleted to make imports succeed.** The four trainers do not
stub out their reward or advantage computation when TRL is missing: `trainers/trl_bridge.py:
require` raises `MissingDependencyError` naming the module, the purpose and the requirement
sheet to read, and `--dry-run` prints the full plan (sampler layout, TRL kwargs, budget,
advantage identity) *without* importing torch or TRL. That is why all four entry points
produce a plan on this machine while none of them can train on it. The alternative —
guarding each method behind `try: import trl` and substituting a placeholder — would have
made the repository import cleanly and prove nothing.

---

## 8. CPU tests — command and actual result

```
$ bash scripts/test_cpu.sh
=== 2026-11-04T23:47:17+08:00 ===
pwd:  /data1/yanghao/paper/GRPO/grpo_baselines
python: Python 3.9.21
executable: /data1/yanghao/miniconda3/envs/judgelrm-sft/bin/python

# 428 progress dots elided. The warning below is verbatim:
# test_rollout_composition triggers a failing logger on purpose, to prove
# a broken diagnostic does not abort training.
warning: rollout log diagnostics failed (RuntimeError: diagnostics exploded); training continues but the flat/underfilled counters for this batch are missing
...
Ran 428 tests

OK
EXIT=0
```

The transcript above is generated from the log of the run that produced the
table below; the elapsed time varies by a second or two between runs and is not
part of any assertion.

| File | Tests | Covers |
|---|---|---|
| `test_adapters.py` | 83 | label algebra, pair views, HelpSteer schema, real-data census, RewardBench isolation, eval views, prompt contracts, TRL rows, length audit, reproducibility, config validation |
| `test_budget_table.py` | 4 | every method row in the budget table answers all eight Sec. 9 questions |
| `test_cli_args.py` | 10 | `--limit` rejection, checkpoint resume validation, output-dir semantics |
| `test_config_prose.py` | 4 | a config's prose cannot contradict the number it configures |
| `test_configs.py` | 39 | shipped configs, derived budget, sampler layout, plan identity, missing dependencies |
| `test_dry_run.py` | 12 | `--limit` narrowing, budget comparability, dry run without TRL |
| `test_evaluation.py` | 42 | prediction schema, accuracy report, both-order vs consistency, official aggregate, calibration, runner |
| `test_j1.py` | 21 | pair scoring, position bias, reward function, real pair identity |
| `test_j4r_advantages.py` | 26 | paper worked example (3 variants), grouping, degenerate groups, state-swap invariance, wrong group sizes, EIS != vanilla |
| `test_jwc_brier.py` | 38 | probability parser, Brier arithmetic, reward direction, invalid output, summary, GT space |
| `test_parsing_and_rewards.py` | 31 | vanilla parser, reward three-state, pinned TRL advantage contract, cross-format parser boundaries |
| `test_pins.py` | 9 | the four recorded commits are the checked-out HEADs, working trees clean, nothing ahead of a pin; the read-only inputs exist and refuse writes |
| `test_report_consistency.py` | 11 | every recomputable number in this report is checked against the code |
| `test_rollout_composition.py` | 5 | MRO ordering: the logger runs after the method hook; a failing log does not abort training |
| `test_rollout_health.py` | 7 | diagnostics reaching `log_history`, manifest declaration, all four methods wrapped |
| `test_rollout_log.py` | 26 | record shape, validity vs reward presence, flat/underfilled diagnostics, wrapping transparency |
| `test_scripts.py` | 11 | launch/eval scripts reject bad arguments, missing model/optimizer state, missing checkpoints and unknown methods before loading anything |
| `test_traceability.py` | 19 | `group_id` / `source_line` / `source_dataset` survive source row -> pair view -> TRL row -> rollout log, and the eight required fields reach the training row, all checked against the source file |
| `test_validation_purity.py` | 6 | validating a config touches nothing on disk |
| `test_verl_adv.py` | 19 | EIS as a verl estimator: hand-worked batch, dtype, one-column read, zero-variance state still carries the global term, uid contract, registry protocol |
| `test_writers.py` | 5 | every shipped writer makes a file, never a directory |

No test was deleted, no answer was hardcoded, and no algorithm was mocked. The
cross-checks that would be easiest to fake are the ones written most strictly: the
paper's three advantage tables are asserted element-wise within `5e-4`;
`PaperExampleTest` separately distinguishes population from sample σ; and
`BothOrderVersusConsistencyTest` includes a judge whose position consistency is `1.0`
while its both-order accuracy is `0.0`, which is exactly the case a combined metric
would hide.

Real defects were found by these tests and the dry runs, and fixed in production code
rather than in the tests. The ones worth naming:

* `write_jsonl` / `write_json` passed a *file* path to `assert_writable_output`, which
  `mkdir`-ed it — creating a directory where a file was expected, so the failure
  surfaced later with a misleading message.
* `rewards/j1.py` let a duplicate `(pair, view)` overwrite an earlier row, leaving that
  row permanently at reward `0.0` — an un-computed reward entering the gradient.
* JwC prediction rows had `outcome = None`, so the only method with a probability head
  could not be scored by any of the accuracy metrics.
* `metrics.aggregate_subsets` crashed on any single-ordering run.
* `--limit` below `global_batch_size` crashed the sampler, and `--dry-run` required TRL.
* `write_manifest` was annotated `-> str` but returned whatever `write_json` returned,
  which is `None`.
* `RolloutLogMixin.log_rollout_health` initially had no caller at all — the diagnostics
  were computed on every batch and then dropped, which is the same as not having them.
* `configs/j4r.yaml` said `kl_beta: 0.01` while its own `controlled_reference` prose claimed
  "KL beta is 1e-4 as in the paper" — a leftover from before the comparison was made
  uniform. `validate_config` checks the number and nothing checks the sentence, so the
  sentence a reader would quote was the wrong one.
* `configs/controlled_budget.yaml` answered four of the eight questions Sec. 9 asks per
  method: it had states, rollouts per state, rollouts per update and steps, but the token
  columns and the frozen-reference decision lived only in the shared block.

---

## 9. Launch and evaluation commands (drafted, **not verified**)

Paths are absolute as the brief requires. Every one of these needs an environment that
does not exist yet, so none has been run to completion.

```bash
# --- Vanilla -------------------------------------------------------------
cd /data1/yanghao/paper/GRPO/grpo_baselines
python -m trainers.vanilla.trl_trainer \
    --config /data1/yanghao/paper/GRPO/grpo_baselines/configs/vanilla.yaml
# -> runs/vanilla/run_manifest.json, runs/vanilla/final/

# --- J1 ------------------------------------------------------------------
python -m trainers.j1.trl_trainer \
    --config /data1/yanghao/paper/GRPO/grpo_baselines/configs/j1.yaml

# --- J4R -----------------------------------------------------------------
python -m trainers.j4r.trl_trainer \
    --config /data1/yanghao/paper/GRPO/grpo_baselines/configs/j4r.yaml
# verl alternative (needs uid = "<pair_id>::<view>"):
#   algorithm.adv_estimator=j4r_eis

# --- JwC -----------------------------------------------------------------
python -m trainers.jwc.trl_trainer \
    --config /data1/yanghao/paper/GRPO/grpo_baselines/configs/jwc_rl_brier.yaml

# --- RewardBench, both orderings, every row ------------------------------
bash /data1/yanghao/paper/GRPO/grpo_baselines/scripts/eval_rewardbench.sh \
     j4r /data1/yanghao/paper/GRPO/grpo_baselines/runs/j4r/final
# -> 2 985 x 2 = 5 970 prediction rows, unparsable rows included,
#    plus <output>.summary.json
```

The only command that executes today is the plan:

```bash
python -m trainers.j4r.trl_trainer --config configs/j4r.yaml --dry-run
```

which prints dataset census, sampler layout, TRL kwargs, budget and advantage identity
as JSON, without importing torch or TRL. Verified for all four methods.

---

## 10. Evaluation interface

`evaluation/rewardbench.py` writes one row per sample per ordering — 5 970 rows for the
full set — and **never filters**. A row whose completion does not parse is written with
`parsed_ok = False` and its `parse_reason`, and the aggregate reports its rate rather
than dropping it. Each of the 22 fields is preserved: `qid`, `view_id`, `source_subset`,
`is_swapped`, `display_order`, `ground_truth`, the rendered `question` / `answer_a` /
`answer_b`, `raw_completion`, parse outcome and reason, `outcome`, `chosen_response`,
`probability`, `probability_decision`, `probability_target`, `correct`,
`latency_seconds`, `method`, `model_path`.

`evaluation/metrics.py` defines, and the tests pin:

* **Original-order accuracy**, reported twice: `accuracy_on_parsed` and
  `accuracy_all_treated_wrong`. The second is there so a method cannot buy accuracy by
  refusing to answer.
* **Parser success rate** and **invalid-output rate**.
* **Both-order accuracy** — `1` only when both orderings are correct.
* **Position consistency** — whether both orderings select the *same physical response*.
  Kept separate from both-order accuracy, and the test suite contains a judge with
  consistency `1.0` and both-order `0.0` (it always picks the worse response) plus one
  with both at `1.0` (position-neutral) to prove the two are not the same number.
* **Brier / ECE / MCE** only for JwC. `calibration_metrics` raises for the hard-verdict
  methods rather than inventing a `1.0` confidence for them, so no hard A/B model can be
  entered into a calibration comparison.
* **23 per-subset accuracies** and the official four-category aggregate, which averages
  within a category and then across categories (not across rows — the two differ, and a
  test pins the difference).

### 10.1 Counting rules, stated as denominators

| Quantity | Numerator | Denominator | Unparsable rows |
|---|---|---|---|
| `parser_success_rate` | rows with `parsed_ok` | all rows in that view | counted as failures |
| `invalid_output_rate` | rows without `parsed_ok` | all rows in that view | they *are* the numerator |
| `accuracy_on_parsed` | correct rows | rows with `parsed_ok` only | **excluded** |
| `accuracy_all_treated_wrong` | correct rows | all rows | counted as wrong |
| `both_order_accuracy` | qids where both orderings are correct | qids with both orderings present | an unparsable side makes the pair incorrect |
| `position_consistency` | qids where both orderings chose the same physical response | qids with both orderings present | an unparsable side counts as *inconsistent*, but stays in the denominator |
| Brier / ECE / MCE | — | rows with a parsable probability | excluded, and reported as `excluded_unparsable` |

Both accuracy columns are always emitted together. Reporting only `accuracy_on_parsed`
would let a model improve its score by declining to answer, which is the failure mode the
second column exists to expose.

**Ties.** A `gt_class` of `TIE` has no correct answer, so `calibration_metrics` skips such
rows and counts them in `skipped_tie` rather than scoring them as wrong. This file contains
none — 2 985 rows, `gt_class` A 1 495 / B 1 490, zero ties — but the rule is implemented
and tested, because it is exactly the kind of branch that rots when it never runs.

Single-ordering runs are supported: the missing aggregate segment reports
`{"available": False, "reason": "…cannot be averaged"}` instead of raising or silently
reporting a number computed from half the data.

---

## 11. Differences from the papers, and open risks

1. **No official training code exists for J1, J4R or JwC.** The J4R organisation
   publishes evaluation code only (`external/reasoning-judge-bench`), which was used for
   the prompt template and the verdict parsing rule. Everything else was re-derived from
   the papers. "Reproduced" would be the wrong word; this is an independent
   implementation of the published objectives.
2. **3B model.** Every paper reports 7B–70B. Absolute scores will not be comparable;
   only differences between the four methods, all trained here under one budget, carry
   information.
3. **Budget is 16 384 rollouts, not the papers'.** J4R uses G=32 per prompt; here 8 per
   state, 16 per pair. With 16 rollouts split across two states, some J4R groups will be
   zero-variance by chance, where EIS correctly returns an advantage of `0` — that is a
   real reduction in learning signal, recorded as an open risk, not a bug.
4. **JwC is a sparse-label adaptation.** The published objective assumes a dense
   preference distribution; HelpSteer3 supplies one hard label. The Brier target is
   therefore `0` or `1`, never a fraction, so the calibration head is trained on a
   strictly harder and less informative signal.
5. **KL is 0.01 in all four methods.** J4R's paper uses `1e-4` and J1's 70B run uses
   `0`; those were deliberately overridden so that KL is not a confound. The deviation
   is recorded in `controlled_budget.notes`.
6. **J1's consistency is estimated within a generation batch.** Both views must be
   present for the bonus, and the guard raises if they are not — but the reward is still
   a batch-local estimate of a pair-level property.
7. **The paper erratum in J4R's Appendix D.1** (the `1.732` that should be `0.285`) is
   documented in `trainers/j4r/eis.py`. If a future reader takes the printed table as
   ground truth, the implementation will look wrong.
8. **TRL API drift.** The J4R injection depends on two private-seeming behaviours of
   `GRPOTrainer` at the pinned commit. The suite reads the pinned file and asserts the
   relevant lines still exist, so an upgrade fails a test instead of producing quietly
   wrong training.
9. **Nothing here has been executed on a GPU.** Import errors, dtype problems, OOM,
   generation-config interactions and the actual `advantages` shape under
   `steps_per_generation > 1` are all unverified in practice.

---

## 12. Status table

| Method | Data adapter | Core code | CPU tests | GPU smoke test | Full training |
|---|---|---|---|---|---|
| Vanilla GRPO | ✅ | ✅ `trainers/vanilla/` | ✅ passed | **Not run** | **Not run** |
| J1 | ✅ (two states) | ✅ `trainers/j1/` + `rewards/j1.py` | ✅ passed | **Not run** | **Not run** |
| J4R | ✅ (two states) | ✅ `trainers/j4r/` (EIS + verifier) | ✅ passed | **Not run** | **Not run** |
| Judging with Confidence (JwC) | ✅ | ✅ `trainers/jwc/` + `rewards/jwc_brier.py` | ✅ passed | **Not run** | **Not run** |
| Evaluation | ✅ (both orderings) | ✅ `evaluation/` | ✅ passed | **Not run** | RewardBench inference **Not run** |

`Ran 428 tests / OK`, Python 3.11.18, no GPU used.

What was authorised was implementation and CPU verification. No training and no model inference were authorised, and none was performed: the launch and evaluation commands in Sec. 9 remain unexecuted drafts until GPU capacity is restored and a new, explicit authorisation is given.
