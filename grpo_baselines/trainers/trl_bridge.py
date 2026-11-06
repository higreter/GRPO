"""Shared configuration bridge between :mod:`common.config` and TRL.

The project config (``configs/*.yaml``) is the single source of truth for the
budget arithmetic that the report quotes.  TRL's ``GRPOConfig`` needs the same
numbers expressed in *its* vocabulary, and the mapping is not one-to-one:

===============================  ==========================================
project field                    TRL field
===============================  ==========================================
``micro_batch_size``             ``per_device_train_batch_size`` (single GPU)
``gradient_accumulation``        ``gradient_accumulation_steps``
``training_steps``               ``max_steps``
``num_generations``              ``num_generations``
``learning_rate``                ``learning_rate``
``kl_beta``                      ``beta``
``clip_epsilon``                 ``epsilon``
``temperature``                  ``temperature``
``top_p``                        ``top_p``
``max_completion_length``        ``max_completion_length``
``global_batch_size`` * states   ``generation_batch_size``
===============================  ==========================================

Two derived constraints are asserted here rather than left to a stack trace at
run time:

1. ``per_device_train_batch_size * gradient_accumulation_steps`` must cover one
   optimiser step's worth of rows (``rows_per_update``).  TRL computes
   ``generation_batch_size = per_device_train_batch_size * world_size *
   steps_per_generation`` when ``steps_per_generation`` is given, so with
   ``steps_per_generation == gradient_accumulation_steps`` the generation batch
   equals one update's row count -- exactly what the config's ``rows_per_update``
   means.
2. ``generation_batch_size % num_generations == 0`` (TRL's ``__post_init__``
   enforces this too, but we want a project-level error message).

``max_prompt_length`` has no TRL counterpart: TRL never truncates prompts
(``_tokenize_prompts`` has no length cap).  The length guard is therefore
enforced upstream by ``data/length_audit.py``; see ``assert_prompt_budget``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from common import paths
from common.config import (
    REFERENCE_POLICY_FROZEN,
    Config,
)

#: Set on the generated ``GRPOConfig`` so a later reader can tell which project
#: config produced a run directory.
RUN_MANIFEST_FILENAME = "run_manifest.json"

#: Flags that every method shares and that no project config field controls.
#: Kept in one dict so the four trainers cannot drift apart: identical values
#: across methods is part of what makes the comparison controlled.
#: ``report_to="none"`` because ``wandb`` is not installed in the target
#: environment and a silent fallback would hide that fact.
#: ``bf16=True`` matches the model's ``torch_dtype``; ``gradient_checkpointing``
#: is required to fit a 3B model plus a frozen reference on 24 GB cards.
DEFAULT_TRAINING_FLAGS = {
    "bf16": True,
    "fp16": False,
    "gradient_checkpointing": True,
    "report_to": "none",
    "dataloader_num_workers": 0,
    "logging_first_step": True,
}

#: Import name -> the requirements file that documents how to obtain it.
INSTALL_HINTS = {
    "torch": "requirements/trl-py311.txt",
    "transformers": "requirements/trl-py311.txt",
    "trl": "requirements/trl-py311.txt",
    "datasets": "requirements/trl-py311.txt",
    "accelerate": "requirements/trl-py311.txt",
    "verl": "requirements/verl-py310.txt",
    "ray": "requirements/verl-py310.txt",
    "vllm": "requirements/verl-py310.txt",
}


class MissingDependencyError(ImportError):
    """Raised when a training-time dependency is absent.

    The message names the requirements file instead of telling the reader to
    ``pip install`` something: this repository ships instructions, never
    executed installs, so copying a guessed package name would be worse than
    failing with the pointer.
    """

    def __init__(self, module: str, purpose: str) -> None:
        hint = INSTALL_HINTS.get(module, "requirements/README.md")
        super().__init__(
            "{0} is required for {1} but is not installed in this environment. "
            "See {2} for the pinned, not-yet-executed installation instructions. "
            "Nothing is installed automatically.".format(module, purpose, hint)
        )
        self.module = module
        self.purpose = purpose
        self.hint = hint


def require(module: str, purpose: str):
    """Import ``module`` or raise :class:`MissingDependencyError`."""
    import importlib

    try:
        return importlib.import_module(module)
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise MissingDependencyError(module, purpose) from error


def assert_prompt_budget(config: Config) -> None:
    """Check the declared prompt budget against the model context window.

    ``common.config`` already validates ``max_prompt_length + max_completion_length
    <= 32768``; this repeats it next to the code that actually launches training
    so the invariant is visible where it matters.  TRL will *not* enforce it: its
    tokeniser path has no truncation, so a prompt longer than
    ``max_prompt_length`` silently trains the model on a longer sequence and
    blows the RoPE window instead of erroring.  The real guard is the audit in
    ``data/length_audit.py``, which must pass before this function is reached.
    """
    total = config.max_prompt_length + config.max_completion_length
    if total > paths.MAX_CONTEXT_TOKENS:
        raise ValueError(
            "max_prompt_length ({0}) + max_completion_length ({1}) = {2} exceeds the "
            "Qwen2.5-3B context window ({3})".format(
                config.max_prompt_length,
                config.max_completion_length,
                total,
                paths.MAX_CONTEXT_TOKENS,
            )
        )


def rows_per_update(config: Config) -> int:
    """Dataset rows consumed by one optimiser step (one row == one prompt state)."""
    return config.global_batch_size * config.states_per_sample


def generation_batch_size(config: Config) -> int:
    """Rollouts produced per optimiser step (TRL's ``generation_batch_size``).

    ``rows_per_update * rollouts_per_state`` rather than
    ``rows_per_update * num_generations``: a dataset row is one prompt *state*, and
    TRL normalises rewards over blocks of ``num_generations`` consecutive rollouts.
    Feeding it ``rollouts_per_state`` therefore makes one block equal one state,
    which is the subgroup the papers normalise within.  The products agree
    (``states_per_sample * rollouts_per_state == num_generations``), so the total
    rollout budget is unchanged.
    """
    return rows_per_update(config) * config.rollouts_per_state


def build_trl_config(config: Config, *, output_dir: Optional[str] = None) -> Dict[str, Any]:
    """Translate a validated project config into ``GRPOConfig`` keyword arguments.

    Returned as a plain dict so this module stays importable without TRL; the
    caller passes it to ``trl.GRPOConfig(**kwargs)``.

    ``steps_per_generation`` is deliberately left implicit (= TRL's
    ``gradient_accumulation_steps``) and ``per_device_train_batch_size`` is
    ``micro_batch_size``, so TRL's own formula reproduces
    ``generation_batch_size`` above.  That is what makes the "one update consumes
    one full set of pairs" invariant hold for the paired methods (J1/J4R).
    """
    assert_prompt_budget(config)
    # TRL derives generation_batch_size = per_device_train_batch_size * world_size *
    # steps_per_generation and steps_per_generation defaults to
    # gradient_accumulation_steps, so the row count per optimiser step is
    # micro_batch_size * gradient_accumulation.  That has to equal rows_per_update,
    # otherwise one update would not see a complete set of pairs and the J1/J4R
    # pairing would silently degrade.
    rolling_per_step = config.micro_batch_size * config["gradient_accumulation"]
    if rolling_per_step != rows_per_update(config):
        raise ValueError(
            "micro_batch_size * gradient_accumulation = {0} but rows_per_update = "
            "global_batch_size * states_per_sample = {1}; one optimiser step must see a "
            "whole set of states (single-process launch).".format(
                rolling_per_step, rows_per_update(config))
        )
    declared_generation = generation_batch_size(config)
    # TRL's ``per_device_train_batch_size`` counts prompts; each prompt state turns
    # into ``rollouts_per_state`` sequences.  Multiply here so the sequences per
    # forward pass are explicit rather than implied.
    per_device_rows = config.micro_batch_size * config.rollouts_per_state
    if per_device_rows * config["gradient_accumulation"] != declared_generation:
        raise ValueError(
            "per_device_train_batch_size * gradient_accumulation = {0} but TRL's "
            "generation_batch_size would be {1}; the sampler and the dataloader would "
            "disagree about how many rollouts one update holds.".format(
                per_device_rows * config["gradient_accumulation"], declared_generation)
        )

    kwargs: Dict[str, Any] = {
        "output_dir": output_dir or config["output_dir"],
        # Prompt states per micro-batch, scaled to sequences: see the check above.
        "per_device_train_batch_size": config.micro_batch_size * config.rollouts_per_state,
        "gradient_accumulation_steps": config["gradient_accumulation"],
        "generation_batch_size": declared_generation,
        # Per-state rollouts, not the per-pair total: see generation_batch_size().
        "num_generations": config.rollouts_per_state,
        "max_completion_length": config.max_completion_length,
        "learning_rate": config["learning_rate"],
        "lr_scheduler_type": config["lr_scheduler_type"],
        "warmup_ratio": config["warmup_ratio"],
        "max_steps": config.training_steps,
        "beta": config.kl_beta,
        "epsilon": config["clip_epsilon"],
        "temperature": config["temperature"],
        "top_p": config["top_p"],
        "seed": config.seed,
        # Keep the reward honest: TRL's default is per-group std scaling, which is
        # exactly GRPO's own normalisation and is what vanilla/J1/JwC want.  J4R
        # overrides the advantages anyway, but setting it explicitly keeps the
        # baseline comparable and documents the intent.
        "scale_rewards": "group",
        "loss_type": "grpo",
        "logging_steps": config["logging"]["log_every_n_steps"],
        "save_steps": config["checkpoint"]["save_every_n_steps"],
        "save_total_limit": config["checkpoint"]["save_total_limit"],
        # This is the actual Transformers switch.  Merely keeping our project-level
        # save_optimizer_state=false in YAML does not stop Trainer._save_checkpoint
        # from writing optimizer.pt and scheduler.pt.
        "save_only_model": not config["checkpoint"]["save_optimizer_state"],
        "log_completions": bool(config["logging"]["save_completion_samples"]),
        # Custom columns must survive: our reward functions need ground_truth,
        # pair_id, sample_id, is_swapped ...
        "remove_unused_columns": False,
    }
    return kwargs


def reference_model_policy_note(config: Config) -> str:
    """Human-readable description of the reference/KL setting for the manifest."""
    if config["reference_model_policy"] == REFERENCE_POLICY_FROZEN:
        return (
            "frozen initial policy as reference; KL coefficient beta={0}".format(
                config.kl_beta)
        )
    return "no reference model; KL term disabled (beta=0)"


def build_manifest(config: Config, *, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run manifest written next to the checkpoints.

    The manifest is what makes a run auditable after the fact: it records the
    resolved budget, the pinned upstream commits and the data fingerprints, so a
    result cannot be reinterpreted later against a different dataset revision.
    """
    from common import reproducibility

    payload: Dict[str, Any] = {
        "method": config.method,
        "output_format": config.output_format,
        "config_source": config.source,
        "base_model": str(paths.BASE_MODEL),
        "train_dataset": str(paths.HELPSTEER_TRAIN),
        "upstream_pins": {name: dict(pin) for name, pin in paths.UPSTREAM_PINS.items()},
        "budget": {
            "global_batch_size": config.global_batch_size,
            "states_per_sample": config.states_per_sample,
            "rows_per_update": config.rows_per_update,
            "num_generations": config.num_generations,
            "rollouts_per_state": config.rollouts_per_state,
            "rollouts_per_update": config.rollouts_per_update,
            "training_steps": config.training_steps,
            "total_rollouts": config["total_rollouts"],
            "estimated_generated_tokens": config["estimated_generated_tokens"],
        },
        "reference": {
            "policy": config["reference_model_policy"],
            "kl_beta": config.kl_beta,
            "note": reference_model_policy_note(config),
        },
        "environment": reproducibility.environment_snapshot(),
    }
    if extra:
        payload["extra"] = extra
    return payload


__all__ = [
    "DEFAULT_TRAINING_FLAGS",
    "INSTALL_HINTS",
    "RUN_MANIFEST_FILENAME",
    "MissingDependencyError",
    "assert_prompt_budget",
    "build_manifest",
    "build_trl_config",
    "generation_batch_size",
    "reference_model_policy_note",
    "require",
    "rows_per_update",
]
