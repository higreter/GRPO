"""Shared TRL wiring for the four baselines.

Every method needs the same five things, and getting any one of them subtly wrong
turns a baseline into a different experiment:

1. **Rows** -- dataset rows built by :func:`data.trl_dataset.build_trl_row`, which
   stores ``prompt`` as a conversation so TRL's chat template is actually applied,
   and carries ``sample_id``/``pair_id``/``view``/``ground_truth`` through to the
   reward function.
2. **A sampler** -- :class:`data.paired_sampler.PairedStateSampler`, which fixes the
   order of the ``original``/``swapped`` states inside one generation batch.  TRL's
   default ``RepeatSampler`` would shuffle rows freely, so a pair's two states could
   land in different updates and J1/J4R would silently degrade to vanilla.
3. **A ``GRPOConfig``** -- :func:`trainers.trl_bridge.build_trl_config`, which is the
   only place the project's batch arithmetic is translated into TRL's.
4. **Reward functions** -- one per method, from :mod:`rewards`.
5. **A manifest** -- the config, the pins, the environment and the verified dataset
   invariants, written next to the checkpoints so a run can be audited later.

This module imports ``trl``/``torch``/``datasets`` **inside** :func:`build_trainer`
only.  Everything else here is importable in the read-only ``judgelrm-sft``
environment, which is what lets the CPU test suite exercise the whole plan --
dataset construction, sampling order, config translation -- without a GPU.

Two things this module deliberately does *not* do:

* it never truncates a prompt (TRL has no ``max_prompt_length``; the length audit in
  :mod:`data.length_audit` is the only line of defence and it fails loudly), and
* it never reshapes a batch to make an error go away.  If the sampler and TRL's
  ``generation_batch_size`` disagree, :func:`trainers.trl_bridge.build_trl_config`
  raises before a single forward pass happens.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from common import config as config_mod
from common import paths, reproducibility
from data import helpsteer_adapter, paired_sampler, trl_dataset

from . import dual_device_reference, trl_bridge

#: Reward-function name -> the sentence the report quotes about it.
REWARD_DESCRIPTIONS: Dict[str, str] = {
    "vanilla": "verdict correctness only (+1 correct, 0 wrong or unparsable)",
    "j1": "verdict correctness (+1 per row) plus verdict consistency (+1 per row of the "
          "pair, awarded only when both orders are judged correctly)",
    "j4r": "judgement reward (+1 correct, 0 otherwise) plus format reward "
           "(+0.5 well-formed, -0.5 malformed), exactly J4R Eq. 6",
    "jwc": "Brier reward 1 - (p - y)^2 on the verbalised probability, 0 when unparsable",
}

#: How each method's advantages are produced, for the manifest and the report.
ADVANTAGE_DESCRIPTIONS: Dict[str, str] = {
    "vanilla": "GRPO group normalisation: (R - mean_state) / (std_state + eps), "
               "one group per prompt state",
    "j1": "GRPO group normalisation per prompt state; the pair-level consistency term "
          "enters through the reward, not through the advantage",
    "j4r": "EIS (J4R Eq. 4): (R - mean_all) / std_all + (R - mean_state) / std_state, "
           "global term across both equivalent states, local term within one state",
    "jwc": "GRPO group normalisation: (R - mean_state) / (std_state + eps), "
           "one group per prompt state",
}


class SamplerMixin(object):
    """Replaces TRL's sampler with the paired one built during planning.

    ``Trainer._get_dataloader`` calls ``sampler_fn(dataset)`` and feeds the result
    straight to ``DataLoader``, so returning a pre-built sampler is enough.  ``__len__``
    of that sampler is ``training_steps * generation_batch_size`` and the dataloader
    batch size is ``generation_batch_size``, so the run yields exactly one generation
    batch per optimiser update.
    """

    _project_sampler = None

    def _get_train_sampler(self, dataset: Any = None) -> Any:
        if self._project_sampler is None:  # pragma: no cover - defensive
            raise RuntimeError(
                "no paired sampler was attached to the trainer; refusing to fall back on "
                "TRL's RepeatSampler because it would reshuffle the linked states"
            )
        return self._project_sampler


# ---------------------------------------------------------------------------
# planning (torch-free)
# ---------------------------------------------------------------------------


class RolloutLogMixin(object):
    """Surface :mod:`rewards.rollout_log` diagnostics as trainer metrics.

    Sec. 5.4 asks for a per-rollout interface; Sec. 5.5 asks that the degenerate cases be
    *handled and recorded*. Handled means "does not produce NaN" (TRL already
    ``nan_to_num``s the advantages). Recorded is the part that needs wiring: without
    this mixin the diagnostics would be computed on every batch and then dropped.

    The metric names are prefixed ``rollout_log/`` so they cannot be confused with a
    reward curve, and the flat *ratio* is logged rather than the flat group count,
    because the count depends on the batch size while the ratio does not.
    """

    def _rollout_log_diagnostics(self):
        """Return the diagnostics of the first reward function that recorded any."""
        for func in getattr(self, "reward_funcs", []) or []:
            target = func
            for attribute in ("module", "_orig_mod"):
                target = getattr(target, attribute, target)
            report = getattr(target, "last_diagnostics", None)
            if report is not None:
                return report
        return None

    def log_rollout_health(self):
        """Push the latest batch's diagnostics into ``self.state.log_history``."""
        report = self._rollout_log_diagnostics()
        if report is None:
            return None
        metrics = {
            "rollout_log/groups": float(report["group_count"]),
            "rollout_log/flat_ratio": float(report["flat_rollout_ratio"]),
            "rollout_log/underfilled": float(report["underfilled_group_count"]),
            "rollout_log/invalid": float(report["invalid"]),
        }
        state = getattr(self, "state", None)
        if state is not None:
            state.log_history.append(metrics)
        return metrics

    def _generate_and_score_completions(self, inputs):
        """Run TRL's generation/scoring step, then record rollout health.

        TRL calls this once per generation batch and slices the result into
        ``steps_per_generation`` gradient-accumulation steps, so this is also the only
        point where the whole group is still in one tensor -- the same boundary
        :func:`rewards.rollout_log.group_diagnostics` assumes.
        """
        output = super()._generate_and_score_completions(inputs)
        try:
            self.log_rollout_health()
        except Exception as error:  # pragma: no cover - defensive by design
            print(
                "warning: rollout log diagnostics failed ({0}: {1}); training continues "
                "but the flat/underfilled counters for this batch are missing".format(
                    type(error).__name__, error),
                file=sys.stderr,
            )
        return output


def load_rows(config: config_mod.Config, *, limit: Optional[int] = None,
              source_row_indices: Optional[Sequence[int]] = None
              ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Materialise TRL rows for a method and verify the pairing invariants."""
    records = list(
        helpsteer_adapter.iter_training_rows(
            two_states=config.states_per_sample == 2, limit=limit,
            source_row_indices=source_row_indices)
    )
    rows = trl_dataset.build_trl_rows(records, config.method)
    summary = trl_dataset.assert_row_invariants(
        rows, states_per_sample=config.states_per_sample)
    return rows, summary


def build_sampler(config: config_mod.Config, rows: Sequence[Dict[str, Any]]
                  ) -> paired_sampler.PairedStateSampler:
    """Build the paired sampler that fixes the state order inside each update.

    A debug run (``--limit``) can hold fewer pairs than ``global_batch_size``.  Rather
    than crash -- the limit switch would then be unusable for exactly the smoke runs
    it exists for -- the batch is narrowed to what exists and the sampler records
    that it was narrowed, so no manifest can claim the declared budget.
    """
    sample_ids = [row["sample_id"] for row in rows]
    layout = paired_sampler.build_pair_layout(
        sample_ids, expected_states=config.states_per_sample)
    declared_pairs = config.global_batch_size
    effective_pairs = min(declared_pairs, len(layout))
    sampler = paired_sampler.PairedStateSampler(
        layout,
        pairs_per_generation=effective_pairs,
        rollouts_per_state=config.rollouts_per_state,
        num_generations=config.num_generations,
        steps=config.training_steps,
        seed=config.seed,
        shuffle=True,
    )
    sampler.narrow_to_layout(declared_pairs)
    if not sampler.batch_narrowed:  # pragma: no cover - defensive
        expected_rows = config.rows_per_update * config.rollouts_per_state
        if sampler.generation_batch_size != expected_rows:
            raise AssertionError(
                "global sampler generation batch {0} != rows_per_update * rollouts_per_state {1}"
                .format(sampler.generation_batch_size, expected_rows))
    return sampler


def plan_run(config: config_mod.Config, *, limit: Optional[int] = None,
             source_row_indices: Optional[Sequence[int]] = None,
             sampler_batches: int = 1) -> Dict[str, Any]:
    """Everything a run would do, described without touching a GPU.

    Used by ``--dry-run``, by the length audit, and by the CPU tests: the tests assert
    on this dictionary rather than on a trainer instance, so "the plan says J4R uses
    EIS" is checked in an environment where TRL is not even installed.
    """
    rows, dataset_summary = load_rows(
        config, limit=limit, source_row_indices=source_row_indices)
    sampler = build_sampler(config, rows)
    trl_kwargs = trl_bridge.build_trl_config(config)
    return {
        "method": config.method,
        "output_format": config.output_format,
        "source": config.source,
        "dataset": dataset_summary,
        "sampler": paired_sampler.describe_sampler(sampler, batches=sampler_batches),
        "reward": {
            "description": REWARD_DESCRIPTIONS[config.method],
        },
        "advantage": {
            "description": ADVANTAGE_DESCRIPTIONS[config.method],
            "evaluated_inside_trainer": config.method == "j4r",
        },
        "trl_config": trl_kwargs,
        "training_flags": dict(trl_bridge.DEFAULT_TRAINING_FLAGS),
        "budget": {
            "states_per_sample": config.states_per_sample,
            "rollouts_per_state": config.rollouts_per_state,
            "rollouts_per_update": config.rollouts_per_update,
            "rows_per_update": config.rows_per_update,
            "training_steps": config.training_steps,
            "total_rollouts": config.total_rollouts_derived,
            "estimated_generated_tokens": config.estimated_generated_tokens_derived,
        },
        "reference": {
            "policy": config["reference_model_policy"],
            "kl_beta": config.kl_beta,
            "note": trl_bridge.reference_model_policy_note(config),
        },
        "debug": {
            "limit": limit,
            "source_row_indices": list(source_row_indices) if source_row_indices is not None else None,
            "batch_narrowed": bool(sampler.batch_narrowed),
            "declared_pairs_per_generation": sampler.declared_pairs_per_generation,
            "effective_pairs_per_generation": sampler.pairs_per_generation,
            "budget_comparable": not sampler.batch_narrowed,
            "note": (
                "the dataset holds fewer pairs than global_batch_size, so this plan "
                "describes a narrowed batch; its token/rollout totals are NOT the "
                "controlled comparison budget"
                if sampler.batch_narrowed else
                "full batch: the budget below is the controlled comparison budget"
            ),
        },
    }


# ---------------------------------------------------------------------------
# training (needs torch/trl)
# ---------------------------------------------------------------------------


def build_trainer(
    config: config_mod.Config,
    *,
    reward_funcs: Sequence[Callable[..., list]],
    rows: Sequence[Dict[str, Any]],
    sampler: paired_sampler.PairedStateSampler,
    trainer_class: Optional[type] = None,
) -> Any:
    """Instantiate TRL's ``GRPOTrainer`` with the project's sampler attached.

    ``trainer_class`` lets the J4R package supply a subclass that overrides
    ``_generate_and_score_completions``; every other method passes ``None`` and gets
    the plain trainer plus :class:`SamplerMixin`.
    """
    trl_bridge.assert_prompt_budget(config)
    trl_bridge.require("torch", "GRPO training")
    trl_bridge.require("trl", "GRPO training")
    trl_bridge.require("datasets", "the training dataset")

    from trl import GRPOConfig, GRPOTrainer

    kwargs = trl_bridge.build_trl_config(config)
    kwargs.update(trl_bridge.DEFAULT_TRAINING_FLAGS)
    args = GRPOConfig(**kwargs)

    dataset = trl_dataset.build_hf_dataset(
        list(rows), purpose="GRPO training rows for method {0!r}".format(config.method))

    base = trainer_class or GRPOTrainer
    topology = dual_device_reference.topology_from_environ()
    if topology is not None:
        if config["reference_model_policy"] != config_mod.REFERENCE_POLICY_FROZEN:
            raise dual_device_reference.DualDeviceReferenceError(
                "dual-device reference requires reference_model_policy=frozen_initial_policy"
            )
        if config.kl_beta <= 0.0:
            raise dual_device_reference.DualDeviceReferenceError(
                "dual-device reference requires kl_beta > 0"
            )
        base = dual_device_reference.wrap_trainer_class(base)

    class _Trainer(SamplerMixin, RolloutLogMixin, base):
        # RolloutLogMixin comes after SamplerMixin and before TRL so its helper methods
        # are available on the instance without shadowing anything TRL defines.
        pass

    _Trainer.__name__ = "{0}Trainer".format(config.method.upper())
    trainer_kwargs = {
        "model": str(paths.BASE_MODEL),
        "reward_funcs": list(reward_funcs),
        "args": args,
        "train_dataset": dataset,
    }
    if topology is not None:
        trainer_kwargs["dual_device_topology"] = topology
    # Load the frozen reference directly on its assigned device.  TRL otherwise
    # constructs it on the policy device before the dual-device mixin can move it.
    # Patching this factory does not alter model weights, the KL objective or the
    # trainer's full-parameter policy: it only avoids a transient second copy.
    if topology is not None:
        import trl.trainer.grpo_trainer as upstream_grpo
        original_create_model = upstream_grpo.create_model_from_path
        reference_device = topology.reference_device

        def create_reference_on_device(model_id, **model_kwargs):
            return original_create_model(model_id, device_map=reference_device, **model_kwargs)

        upstream_grpo.create_model_from_path = create_reference_on_device
        try:
            trainer = _Trainer(**trainer_kwargs)
        finally:
            upstream_grpo.create_model_from_path = original_create_model
    else:
        trainer = _Trainer(**trainer_kwargs)
    trainer._project_sampler = sampler
    return trainer


def write_manifest(config: config_mod.Config, *, extra: Optional[Dict[str, Any]] = None
                   ) -> str:
    """Write ``run_manifest.json`` into the configured output directory."""
    output_dir = str(config["output_dir"])
    paths.assert_writable_output(output_dir)
    manifest_extra = dict(extra or {})
    topology = dual_device_reference.topology_from_environ()
    if topology is not None:
        manifest_extra["dual_device_reference"] = topology.manifest()
    manifest = trl_bridge.build_manifest(config, extra=manifest_extra)
    manifest["reward"] = {"description": REWARD_DESCRIPTIONS[config.method]}
    manifest["advantage"] = {
        "description": ADVANTAGE_DESCRIPTIONS[config.method],
        "evaluated_inside_trainer": config.method == "j4r",
    }
    manifest["rollout_log"] = {
        "enabled": True,
        "module": "rewards.rollout_log",
        "per_rollout_fields": ["prompt_id", "pair_id", "view", "rollout_index",
                               "completion", "reward", "reason", "outcome",
                               "format_ok", "group_key"],
        "grouping": "sample_id ('<pair_id>::<view>') -- the same group TRL normalises "
                    "within for vanilla/J1/JwC, and the local term's group for J4R",
        "diagnostics": ["flat_groups", "flat_rollout_ratio", "underfilled_groups",
                        "incomplete_groups", "invalid"],
        "metrics_prefix": "rollout_log/",
        "retention": "one generation batch in memory; written to disk only when "
                     "logging.save_completion_samples is true",
    }
    target = os.path.join(output_dir, trl_bridge.RUN_MANIFEST_FILENAME)
    reproducibility.write_json(target, manifest)
    return target


def train(
    config: config_mod.Config,
    *,
    reward_funcs: Sequence[Callable[..., list]],
    trainer_class: Optional[type] = None,
    limit: Optional[int] = None,
    source_row_indices: Optional[Sequence[int]] = None,
    dry_run: bool = False,
    resume_from_checkpoint: Optional[str] = None,
) -> Dict[str, Any]:
    """Plan, record and launch one run.

    ``dry_run`` returns the plan and writes nothing, which is what the CPU test suite
    and the shell scripts' ``--dry-run`` use.
    """
    rows, dataset_summary = load_rows(
        config, limit=limit, source_row_indices=source_row_indices)
    sampler = build_sampler(config, rows)
    plan = plan_run(config, limit=limit, source_row_indices=source_row_indices)
    if dry_run:
        plan["dataset"] = dataset_summary
        return plan

    manifest_path = write_manifest(
        config,
        extra={
            "dataset": dataset_summary,
            "sampler": paired_sampler.describe_sampler(sampler, batches=1),
            "plan_budget": plan["budget"],
        },
    )
    reproducibility.seed_everything(config.seed)
    trainer = build_trainer(
        config,
        reward_funcs=reward_funcs,
        rows=rows,
        sampler=sampler,
        trainer_class=trainer_class,
    )
    topology_status = getattr(trainer, "dual_device_reference_status", None)
    if topology_status is not None:
        # Rewrite the already-created manifest with the verified placement status, not
        # merely the requested environment variables.
        manifest = trl_bridge.build_manifest(
            config,
            extra={
                "dataset": dataset_summary,
                "sampler": paired_sampler.describe_sampler(sampler, batches=1),
                "plan_budget": plan["budget"],
                "dual_device_reference": topology_status,
            },
        )
        manifest["reward"] = {"description": REWARD_DESCRIPTIONS[config.method]}
        manifest["advantage"] = {
            "description": ADVANTAGE_DESCRIPTIONS[config.method],
            "evaluated_inside_trainer": config.method == "j4r",
        }
        reproducibility.write_json(manifest_path, manifest)
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    final_dir = os.path.join(str(config["output_dir"]), "final")
    trainer.save_model(final_dir)
    # save_model() writes only weights/config/tokenizer. Preserve the finished
    # metrics next to the final model, but never imply that optimizer state exists.
    trainer.state.save_to_json(os.path.join(final_dir, "trainer_state.json"))
    return {"manifest": manifest_path, "output_dir": str(config["output_dir"])}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class ArgumentError(ValueError):
    """Raised when a command-line value cannot mean what the caller asked for."""


def validate_args(args) -> None:
    """Reject argument values that would otherwise fail late and obscurely.

    Each check exists because the un-validated value produced a *plausible but wrong*
    run rather than an error:

    * ``--limit 0`` or a negative limit selects no pairs, and the sampler would then be
      built over an empty layout;
    * ``--limit`` larger than the dataset looks like a full run in the log while being
      indistinguishable from one that silently dropped rows;
    * a resume directory that does not exist fails inside ``Trainer.train`` with a
      message about a missing file rather than about the flag that was wrong.
    """
    if args.limit is not None:
        if args.limit < 1:
            raise ArgumentError(
                "--limit must be >= 1, got {0}; a limit of zero selects no pairs and "
                "the sampler would have nothing to iterate over".format(args.limit))
    if args.source_row_index is not None:
        if any(index < 0 for index in args.source_row_index):
            raise ArgumentError("--source-row-index values must be >= 0")
        if len(set(args.source_row_index)) != len(args.source_row_index):
            raise ArgumentError("--source-row-index values must be distinct")
    if args.resume_from_checkpoint is not None:
        target = Path(args.resume_from_checkpoint)
        if not target.exists():
            raise ArgumentError(
                "--resume-from-checkpoint {0} does not exist. Point it at a directory "
                "written by trainer.train() (it contains trainer_state.json); a resume "
                "from a missing path would otherwise surface as an unrelated error "
                "inside transformers.".format(target))
        if not target.is_dir():
            raise ArgumentError(
                "--resume-from-checkpoint {0} is not a directory; pass the checkpoint "
                "directory itself, not a file inside it".format(target))
        if not (target / "trainer_state.json").exists():
            raise ArgumentError(
                "--resume-from-checkpoint {0} has no trainer_state.json, so the "
                "optimizer step and the sampler position cannot be restored".format(
                    target))
        if (target / "trainer_state.json").is_file() and not (target / "optimizer.pt").exists():
            raise ArgumentError(
                "--resume-from-checkpoint {0} has no optimizer.pt. These runs save "
                "model-only checkpoints; refusing to silently restart optimisation "
                "from step zero or claim a resumable training state.".format(target))
    if args.dry_run and args.resume_from_checkpoint is not None:
        # Harmless, but worth saying: a dry run never reads the checkpoint, so a user
        # checking a resume path would get a green plan that proves nothing about it.
        print("note: --dry-run ignores --resume-from-checkpoint; the checkpoint is "
              "only validated, never loaded.", file=sys.stderr)


def build_arg_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, help="path to a configs/*.yaml file")
    parser.add_argument("--limit", type=int, default=None,
                        help="use only the first N source pairs (debugging only)")
    parser.add_argument("--source-row-index", type=int, action="append", default=None,
                        help="select a real source-row index for smoke testing; may repeat")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the plan as JSON and exit without training")
    parser.add_argument("--resume-from-checkpoint", default=None,
                        help="checkpoint directory to resume from; it must contain "
                             "trainer_state.json")
    return parser


def run_cli(
    description: str,
    *,
    reward_funcs_factory: Callable[[config_mod.Config], Sequence[Callable[..., list]]],
    trainer_class_factory: Optional[Callable[[config_mod.Config], Optional[type]]] = None,
    argv: Optional[Sequence[str]] = None,
) -> int:
    """Common ``main`` for the four ``python -m trainers.<method>.trl_trainer`` entry points."""
    parser = build_arg_parser(description)
    args = parser.parse_args(argv)
    validate_args(args)
    config = config_mod.load_config(args.config)
    reward_funcs = reward_funcs_factory(config)
    # A dry run must work in an environment without TRL: the plan already states
    # which advantage estimator would run, so no trainer class is needed.
    trainer_class = None
    if trainer_class_factory is not None and not args.dry_run:
        trainer_class = trainer_class_factory(config)
    result = train(
        config,
        reward_funcs=reward_funcs,
        trainer_class=trainer_class,
        limit=args.limit,
        source_row_indices=args.source_row_index,
        dry_run=args.dry_run,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


__all__ = [
    "ADVANTAGE_DESCRIPTIONS",
    "ArgumentError",
    "REWARD_DESCRIPTIONS",
    "RolloutLogMixin",
    "SamplerMixin",
    "build_arg_parser",
    "build_sampler",
    "build_trainer",
    "load_rows",
    "plan_run",
    "run_cli",
    "train",
    "validate_args",
    "write_manifest",
]
