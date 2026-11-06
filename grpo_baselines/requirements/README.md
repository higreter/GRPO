# Requirements: what is pinned, what is not, and why nothing was installed

Three things are being separated here on purpose, because conflating them is how a
"reproduction" quietly becomes a different experiment:

1. **What the CPU tests actually ran on** — the read-only conda env
   `/data1/yanghao/miniconda3/envs/judgelrm-sft`, Python 3.9.21. Nothing was installed
   into it, upgraded, or removed.
2. **What a training run would need** — `trl-py311.txt` (TRL) and `verl-py310.txt`
   (verl). Both are *instruction sheets*: no command in either was executed.
3. **What is missing from the read-only env** — see the table below. The absence is a
   measured fact (`importlib.util.find_spec`), not an assumption.

## Measured state of the read-only environment

| package | state |
| --- | --- |
| python | 3.9.21 |
| torch | 2.8.0+cu128 |
| transformers | 4.57.6 |
| tokenizers | 0.22.2 |
| accelerate | 1.10.1 |
| deepspeed | 0.19.7 |
| safetensors | 0.7.0 |
| numpy | 2.0.2 |
| pyyaml | 6.0.3 |
| jinja2 | 3.1.6 |
| **trl** | **absent** |
| **verl** | **absent** |
| **vllm** | **absent** |
| **ray** | **absent** |
| **datasets** | **absent** |
| **peft** | **absent** |
| **flash_attn** | **absent** |
| **wandb** | **absent** |
| **pytest** | **absent** |

Consequences that shaped the code:

* The whole CPU suite is standard-library `unittest`
  (`python -m unittest discover -s tests -t . -v`). No test needs pytest.
* `torch` / `transformers` / `trl` / `datasets` are imported **lazily, inside functions**,
  so importing the project does not require them. `trainers/trl_bridge.py:require()`
  turns a missing training dependency into a `MissingDependencyError` that names the
  requirements file instead of a bare `ModuleNotFoundError`.
* Only `data/length_audit.py` and `evaluation/rewardbench.py:make_transformers_generator`
  touch `transformers`, and neither is imported by the trainer packages.

## Why two Python versions

Read off the pinned upstream sources:

* `external/trl/pyproject.toml`: `requires-python = ">=3.11"`.
* `external/verl/pyproject.toml`: `requires-python = ">=3.10,<3.13"`.

The window where both exist is **3.11 and 3.12 only**. The two files are named after the
floor each stack allows, not after a tested combination — no combination was tested.

## Why three environments, not one

`external/trl/pyproject.toml` wants `torch>=2.6.0` and `transformers>=4.56.2`.
`external/verl/setup.py` pins `torch==2.9.1`, while its `vllm` extra pins
`torch==2.13.0` / `transformers==5.16.1`. A single `pip install` of both resolves to
whichever was installed last, and the failure mode is not a clean import error — it is a
binary extension built against a different `libtorch`/`libcudart`.

| environment | purpose | never run |
| --- | --- | --- |
| `grpo-trl-py311` | CPU tests, TRL training of the four methods | yes |
| `grpo-verl-py310` | verl rollout + J4R `j4r_eis` advantage estimator | yes |
| judge-eval | RewardBench inference from a checkpoint | yes |

## Named risks (none of them resolved in this task)

1. **vLLM / torch / CUDA ABI.** vLLM wheels are compiled against one torch build and one
   CUDA major version. The RTX 4090 is sm_89 with a CUDA 12.x driver; vendored vLLM 0.32
   targets cu130 / torch 2.13. Mixing it with torch 2.9.1 produces either an import
   failure or, worse, a silently wrong kernel.
2. **FlashAttention.** `flash-attn` is compiled, not interpreted. It must be built after
   torch is frozen, with the same CUDA toolchain. A mismatched wheel is the classic source
   of "trains fine, produces garbage".
3. **NCCL.** Multi-GPU runs need NCCL matching the driver and the CUDA runtime; verl's
   `ray.util.collective` path additionally pulls `cupy-cuda13x`. None of this was exercised:
   the three GPUs on this host were 100% busy with someone else's jobs, so not even a
   one-step smoke test was attempted.
4. **PyTorch pin drift.** This machine's read-only env already has torch 2.8.0+cu128 while
   verl asks for 2.9.1 and vLLM for 2.13.0. Any of the three is a different experiment from
   the others; the report records this rather than pretending one pin covers all.
5. **TRL API drift.** The trainers subclass `GRPOTrainer` and rely on
   `_generate_and_score_completions` returning a dict with an `"advantages"` key
   (`external/trl/trl/trainer/grpo_trainer.py:3037-3060`) and on
   `_calculate_rewards` passing every non-prompt dataset column through as a keyword
   argument (`:1648-1660`). Both are verified against the pinned commit only; a newer TRL
   may move them.

## Commands that were deliberately NOT run

```
conda create -n grpo-trl-py311 python=3.11 -y
pip install -r requirements/trl-py311.txt
pip install "trl @ git+https://github.com/huggingface/trl.git@3885117d6a1dd647d43befeddf0da482c505cb47"
conda create -n grpo-verl-py310 python=3.10 -y
pip install -r requirements/verl-py310.txt
pip install "verl @ git+https://github.com/volcengine/verl.git@745432653d15af14954af1e055c8aeed72afa064"
```

The one command that *was* run, and is reproducible right now:

```
source /data1/yanghao/miniconda3/etc/profile.d/conda.sh && conda activate judgelrm-sft
python -m unittest discover -s tests -t . -v
```
