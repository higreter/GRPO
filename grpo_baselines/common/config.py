"""Loading, validating and resolving the per-method training configs.

A config is a YAML file under ``configs/``.  ``load_config`` parses it, checks every
field that could silently corrupt an experiment, and returns a :class:`Config` whose
derived quantities (rows per update, rollouts per update, estimated generated tokens)
are recomputed from first principles and compared against the values written in the
file.  A mismatch is an error, not a warning: the budget table in the report is only
meaningful if those numbers cannot drift away from the arithmetic that produced them.

Terminology (kept identical in every config, the budget table and the trainers)
------------------------------------------------------------------------------
``global_batch_size``
    Number of **source preference pairs** per optimizer update.
``states_per_sample``
    Number of linked prompt *states* derived from one source pair.  Vanilla/JwC use one
    state; J1 and J4R use two (the original and the swapped order), which is exactly the
    ``T_1(q) .. T_L(q)`` construction of arXiv:2505.13346 Sec. 3.2.
``num_generations``
    Total completions sampled per source pair per update, across all of its states.
``rollouts_per_state``
    ``num_generations // states_per_sample`` -- the group size a state's own
    normalisation is computed over.
``rows_per_update``
    ``global_batch_size * states_per_sample`` -- the number of *dataset rows* (one row is
    one state) that make up an update, i.e. the quantity TRL sees as its batch.

Consequences worth stating out loud: matching ``training_steps`` across methods does not
match compute, because J1/J4R spend twice the rollouts per source pair.  Fairness is
established on ``total_rollouts`` and ``estimated_generated_tokens`` instead, which is
why both are mandatory fields here.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

from . import paths, prompts

#: Fields every method config must define.
REQUIRED_FIELDS: Tuple[str, ...] = (
    "method",
    "output_format",
    "base_model",
    "train_dataset",
    "output_dir",
    "seed",
    "max_prompt_length",
    "max_completion_length",
    "learning_rate",
    "lr_scheduler_type",
    "warmup_ratio",
    "global_batch_size",
    "micro_batch_size",
    "gradient_accumulation",
    "states_per_sample",
    "num_generations",
    "training_steps",
    "total_rollouts",
    "estimated_prompt_tokens",
    "estimated_generated_tokens",
    "temperature",
    "top_p",
    "clip_epsilon",
    "reference_model_policy",
    "kl_beta",
    "logging",
    "checkpoint",
    "controlled_budget",
)

#: Allowed values for ``reference_model_policy``.
REFERENCE_POLICY_FROZEN = "frozen_initial_policy"
REFERENCE_POLICY_NONE = "none"
VALID_REFERENCE_POLICIES: Tuple[str, ...] = (REFERENCE_POLICY_FROZEN, REFERENCE_POLICY_NONE)

#: Methods whose definition requires two linked prompt states.
TWO_STATE_METHODS: Tuple[str, ...] = ("j1", "j4r")

_LOGGING_KEYS = ("log_every_n_steps", "save_completion_samples", "metrics")
_CHECKPOINT_KEYS = ("save_every_n_steps", "save_total_limit", "save_optimizer_state")
_BUDGET_KEYS = (
    "paper_reference",
    "controlled_reference",
    "paper_states_per_sample",
    "paper_rollouts_per_state",
    "paper_rollouts_per_update",
    "paper_training_steps",
    "paper_generated_tokens",
    "notes",
)


class ConfigError(ValueError):
    """Raised when a config is missing a field or contradicts its own arithmetic."""


def _require_int(name: str, value: Any, minimum: Optional[int] = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError("{0} must be an int, got {1!r}".format(name, value))
    if minimum is not None and value < minimum:
        raise ConfigError("{0} must be >= {1}, got {2}".format(name, minimum, value))
    return value


def _require_float(name: str, value: Any, *, low: Optional[float] = None,
                   high: Optional[float] = None, low_inclusive: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError("{0} must be a number, got {1!r}".format(name, value))
    number = float(value)
    if low is not None:
        if low_inclusive and number < low:
            raise ConfigError("{0} must be >= {1}, got {2}".format(name, low, number))
        if not low_inclusive and number <= low:
            raise ConfigError("{0} must be > {1}, got {2}".format(name, low, number))
    if high is not None and number > high:
        raise ConfigError("{0} must be <= {1}, got {2}".format(name, high, number))
    return number


def _require_mapping(name: str, value: Any, keys: Tuple[str, ...]) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError("{0} must be a mapping, got {1!r}".format(name, value))
    missing = [key for key in keys if key not in value]
    if missing:
        raise ConfigError("{0} is missing {1}".format(name, missing))
    unknown = [key for key in value if key not in keys]
    if unknown:
        raise ConfigError("{0} has unknown keys {1}".format(name, unknown))
    return dict(value)


class Config(object):
    """A validated method config plus its derived budget arithmetic."""

    def __init__(self, data: Dict[str, Any], source: Optional[str] = None) -> None:
        self._data = dict(data)
        self.source = source

    # -- field access ------------------------------------------------------------
    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def __contains__(self, key: str) -> bool:
        return key in self._data

    @property
    def method(self) -> str:
        return self._data["method"]

    @property
    def output_format(self) -> str:
        return self._data["output_format"]

    @property
    def seed(self) -> int:
        return self._data["seed"]

    @property
    def max_prompt_length(self) -> int:
        return self._data["max_prompt_length"]

    @property
    def max_completion_length(self) -> int:
        return self._data["max_completion_length"]

    @property
    def states_per_sample(self) -> int:
        return self._data["states_per_sample"]

    @property
    def num_generations(self) -> int:
        return self._data["num_generations"]

    @property
    def global_batch_size(self) -> int:
        return self._data["global_batch_size"]

    @property
    def micro_batch_size(self) -> int:
        return self._data["micro_batch_size"]

    @property
    def gradient_accumulation(self) -> int:
        return self._data["gradient_accumulation"]

    @property
    def training_steps(self) -> int:
        return self._data["training_steps"]

    @property
    def kl_beta(self) -> float:
        return float(self._data["kl_beta"])

    # -- derived budget arithmetic ----------------------------------------------
    @property
    def rollouts_per_state(self) -> int:
        return self.num_generations // self.states_per_sample

    @property
    def rows_per_update(self) -> int:
        return self.global_batch_size * self.states_per_sample

    @property
    def rollouts_per_update(self) -> int:
        return self.global_batch_size * self.num_generations

    @property
    def total_rollouts_derived(self) -> int:
        return self.rollouts_per_update * self.training_steps

    @property
    def estimated_generated_tokens_derived(self) -> int:
        per_rollout = self._data["estimated_prompt_tokens"] + self.max_completion_length
        return self.total_rollouts_derived * per_rollout

    @property
    def context_budget(self) -> int:
        return self.max_prompt_length + self.max_completion_length

    def to_dict(self) -> Dict[str, Any]:
        """Return the raw config plus every derived quantity.

        The extra keys are what the trainer and the report consume, so a reader never has
        to redo the arithmetic in their head.
        """
        payload = dict(self._data)
        payload.update(
            {
                "derived": {
                    "rollouts_per_state": self.rollouts_per_state,
                    "rows_per_update": self.rows_per_update,
                    "rollouts_per_update": self.rollouts_per_update,
                    "total_rollouts": self.total_rollouts_derived,
                    "estimated_generated_tokens": self.estimated_generated_tokens_derived,
                    "context_budget": self.context_budget,
                    "context_budget_limit": paths.MAX_CONTEXT_TOKENS,
                    "source": self.source,
                }
            }
        )
        return payload

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "Config(method={0!r}, steps={1}, rollouts={2})".format(
            self.method, self.training_steps, self.total_rollouts_derived
        )


def validate_config(data: Dict[str, Any], source: Optional[str] = None) -> Config:
    """Validate a raw mapping and return a :class:`Config`."""
    if not isinstance(data, dict):
        raise ConfigError("config must be a mapping, got {0!r}".format(type(data).__name__))

    missing = [field for field in REQUIRED_FIELDS if field not in data]
    if missing:
        raise ConfigError("missing required fields: {0}".format(missing))
    unknown = [key for key in data if key not in REQUIRED_FIELDS]
    if unknown:
        raise ConfigError("unknown fields (typo?): {0}".format(unknown))

    method = data["method"]
    if method not in prompts.VALID_METHODS:
        raise ConfigError("method must be one of {0}, got {1!r}".format(
            prompts.VALID_METHODS, method))
    output_format = data["output_format"]
    if output_format not in prompts.VALID_OUTPUT_FORMATS:
        raise ConfigError("output_format must be one of {0}, got {1!r}".format(
            prompts.VALID_OUTPUT_FORMATS, output_format))
    if output_format != method:
        raise ConfigError(
            "output_format must match method (got {0!r} for method {1!r}); the parser is "
            "selected by this field, so a mismatch would score one method with another's "
            "reward".format(output_format, method)
        )

    # -- fixed assets ------------------------------------------------------------
    if os.path.realpath(str(data["base_model"])) != os.path.realpath(str(paths.BASE_MODEL)):
        raise ConfigError(
            "base_model must be the fixed project model {0}, got {1!r}".format(
                paths.BASE_MODEL, data["base_model"])
        )
    train_dataset = os.path.realpath(str(data["train_dataset"]))
    if train_dataset != os.path.realpath(str(paths.HELPSTEER_TRAIN)):
        raise ConfigError(
            "train_dataset must be the fixed training source {0}, got {1!r}. RewardBench "
            "must never become a training file.".format(paths.HELPSTEER_TRAIN,
                                                        data["train_dataset"])
        )

    # check, do not create: validating a yaml must leave the filesystem alone
    paths.check_writable_output(data["output_dir"])

    # -- numeric sanity ----------------------------------------------------------
    _require_int("seed", data["seed"], minimum=0)
    max_prompt = _require_int("max_prompt_length", data["max_prompt_length"], minimum=1)
    max_completion = _require_int(
        "max_completion_length", data["max_completion_length"], minimum=1)
    if max_prompt + max_completion > paths.MAX_CONTEXT_TOKENS:
        raise ConfigError(
            "max_prompt_length + max_completion_length = {0} exceeds the model context "
            "window {1}".format(max_prompt + max_completion, paths.MAX_CONTEXT_TOKENS)
        )

    _require_float("learning_rate", data["learning_rate"], low=0.0, low_inclusive=False)
    _require_float("warmup_ratio", data["warmup_ratio"], low=0.0, high=1.0)
    temperature = _require_float("temperature", data["temperature"], low=0.0,
                                 low_inclusive=False)
    _require_float("top_p", data["top_p"], low=0.0, high=1.0, low_inclusive=False)
    _require_float("clip_epsilon", data["clip_epsilon"], low=0.0, low_inclusive=False)
    kl_beta = _require_float("kl_beta", data["kl_beta"], low=0.0)
    if data["lr_scheduler_type"] not in ("constant", "linear", "cosine"):
        raise ConfigError("lr_scheduler_type must be constant/linear/cosine, got {0!r}".format(
            data["lr_scheduler_type"]))

    # -- group structure ---------------------------------------------------------
    states = _require_int("states_per_sample", data["states_per_sample"], minimum=1)
    expected_states = 2 if method in TWO_STATE_METHODS else 1
    if states != expected_states:
        raise ConfigError(
            "method {0!r} requires states_per_sample={1}, got {2}".format(
                method, expected_states, states)
        )
    generations = _require_int("num_generations", data["num_generations"], minimum=2)
    if generations % states != 0:
        raise ConfigError(
            "num_generations={0} is not divisible by states_per_sample={1}".format(
                generations, states)
        )
    if generations // states < 2:
        raise ConfigError(
            "rollouts_per_state={0} < 2: GRPO cannot normalise a group of size 1".format(
                generations // states)
        )

    global_batch = _require_int("global_batch_size", data["global_batch_size"], minimum=1)
    micro_batch = _require_int("micro_batch_size", data["micro_batch_size"], minimum=1)
    accumulation = _require_int("gradient_accumulation", data["gradient_accumulation"],
                                minimum=1)
    rows_per_update = global_batch * states
    if rows_per_update % micro_batch != 0:
        raise ConfigError(
            "rows_per_update={0} (global_batch_size * states_per_sample) is not divisible "
            "by micro_batch_size={1}".format(rows_per_update, micro_batch)
        )
    if accumulation != rows_per_update // micro_batch:
        raise ConfigError(
            "gradient_accumulation={0} contradicts rows_per_update/micro_batch_size={1}".format(
                accumulation, rows_per_update // micro_batch)
        )

    steps = _require_int("training_steps", data["training_steps"], minimum=1)
    total_rollouts = _require_int("total_rollouts", data["total_rollouts"], minimum=1)
    expected_rollouts = steps * global_batch * generations
    if total_rollouts != expected_rollouts:
        raise ConfigError(
            "total_rollouts={0} but training_steps * global_batch_size * num_generations "
            "= {1}".format(total_rollouts, expected_rollouts)
        )

    estimated_prompt = _require_int("estimated_prompt_tokens", data["estimated_prompt_tokens"],
                                    minimum=1)
    estimated_tokens = _require_int("estimated_generated_tokens",
                                    data["estimated_generated_tokens"], minimum=1)
    expected_tokens = total_rollouts * (estimated_prompt + max_completion)
    if estimated_tokens != expected_tokens:
        raise ConfigError(
            "estimated_generated_tokens={0} but total_rollouts * (estimated_prompt_tokens + "
            "max_completion_length) = {1}".format(estimated_tokens, expected_tokens)
        )

    # -- optimiser reference policy ----------------------------------------------
    policy = data["reference_model_policy"]
    if policy not in VALID_REFERENCE_POLICIES:
        raise ConfigError("reference_model_policy must be one of {0}, got {1!r}".format(
            VALID_REFERENCE_POLICIES, policy))
    if policy == REFERENCE_POLICY_FROZEN and kl_beta <= 0.0:
        raise ConfigError("reference_model_policy=frozen_initial_policy requires kl_beta > 0")
    if policy == REFERENCE_POLICY_NONE and kl_beta != 0.0:
        raise ConfigError("reference_model_policy=none requires kl_beta == 0")

    # -- nesting -----------------------------------------------------------------
    logging_cfg = _require_mapping("logging", data["logging"], _LOGGING_KEYS)
    _require_int("logging.log_every_n_steps", logging_cfg["log_every_n_steps"], minimum=1)
    if not isinstance(logging_cfg["save_completion_samples"], bool):
        raise ConfigError("logging.save_completion_samples must be a bool")
    if not isinstance(logging_cfg["metrics"], list) or not logging_cfg["metrics"]:
        raise ConfigError("logging.metrics must be a non-empty list")

    checkpoint_cfg = _require_mapping("checkpoint", data["checkpoint"], _CHECKPOINT_KEYS)
    _require_int("checkpoint.save_every_n_steps", checkpoint_cfg["save_every_n_steps"],
                 minimum=1)
    _require_int("checkpoint.save_total_limit", checkpoint_cfg["save_total_limit"], minimum=1)
    if not isinstance(checkpoint_cfg["save_optimizer_state"], bool):
        raise ConfigError("checkpoint.save_optimizer_state must be a bool")

    budget_cfg = _require_mapping("controlled_budget", data["controlled_budget"], _BUDGET_KEYS)
    for key in ("paper_reference", "controlled_reference", "notes"):
        if not isinstance(budget_cfg[key], str) or not budget_cfg[key].strip():
            raise ConfigError("controlled_budget.{0} must be a non-empty string".format(key))
    for key in ("paper_states_per_sample", "paper_rollouts_per_state",
                "paper_rollouts_per_update", "paper_training_steps", "paper_generated_tokens"):
        _require_int("controlled_budget.{0}".format(key), budget_cfg[key], minimum=1)

    del temperature  # validated above; kept in the mapping
    return Config(data, source=source)


def load_config(path: str) -> Config:
    """Read a YAML config from disk, validate it and return the resolved object."""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - PyYAML is present in the audit env
        raise ConfigError(
            "PyYAML is required to read configs ({0}); it ships with the project "
            "environment".format(exc)
        )
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        raise ConfigError("{0}: file is empty".format(path))

    # The repository is designed to be moved as one GRPO directory.  Dataset and output
    # paths in the shipped YAML files are therefore relative to ``grpo_baselines`` and
    # are made absolute here before validation.  The base weights are intentionally not
    # part of the transfer bundle; ``GRPO_BASE_MODEL`` points to their location on the
    # destination server and overrides the provenance path recorded in the YAML.
    for field in ("train_dataset", "output_dir"):
        value = os.path.expanduser(str(data[field]))
        if not os.path.isabs(value):
            value = os.path.join(str(paths.PROJECT_ROOT), value)
        data[field] = os.path.abspath(value)
    if os.environ.get("GRPO_BASE_MODEL"):
        data["base_model"] = str(paths.BASE_MODEL)
    else:
        value = os.path.expanduser(str(data["base_model"]))
        if not os.path.isabs(value):
            value = os.path.join(str(paths.PROJECT_ROOT), value)
        data["base_model"] = os.path.abspath(value)
    return validate_config(data, source=os.path.abspath(path))


def load_all_configs(config_dir: Optional[str] = None) -> Dict[str, Config]:
    """Load every ``configs/*.yaml`` except the budget table, keyed by method."""
    directory = config_dir or str(paths.PROJECT_ROOT / "configs")
    configs: Dict[str, Config] = {}
    for name in sorted(os.listdir(directory)):
        if not name.endswith(".yaml") or name.startswith("controlled_budget"):
            continue
        config = load_config(os.path.join(directory, name))
        if config.method in configs:
            raise ConfigError("duplicate config for method {0!r}".format(config.method))
        configs[config.method] = config
    if not configs:
        raise ConfigError("no method configs found under {0}".format(directory))
    return configs


def assert_equal_budget(configs: Dict[str, Config], keys: Tuple[str, ...] = (
    "total_rollouts", "estimated_generated_tokens",
)) -> List[str]:
    """Return a human-readable report of the controlled-budget agreement.

    Raises when the methods disagree on a budget key: the whole point of the controlled
    comparison is that the *token* budget matches even though the number of optimizer
    steps does not.
    """
    if not configs:
        raise ConfigError("no configs supplied")
    problems: List[str] = []
    reference = None
    for method in sorted(configs):
        config = configs[method]
        values = {
            "total_rollouts": config.total_rollouts_derived,
            "estimated_generated_tokens": config.estimated_generated_tokens_derived,
        }
        if reference is None:
            reference = (method, values)
            continue
        for key in keys:
            if values[key] != reference[1][key]:
                problems.append(
                    "{0}.{1}={2} differs from {3}.{1}={4}".format(
                        method, key, values[key], reference[0], reference[1][key])
                )
    if problems:
        raise ConfigError("controlled budget mismatch: " + "; ".join(problems))
    return ["{0}: {1}".format(key, reference[1][key]) for key in keys]


__all__ = [
    "Config",
    "ConfigError",
    "REFERENCE_POLICY_FROZEN",
    "REFERENCE_POLICY_NONE",
    "REQUIRED_FIELDS",
    "TWO_STATE_METHODS",
    "VALID_REFERENCE_POLICIES",
    "assert_equal_budget",
    "load_all_configs",
    "load_config",
    "validate_config",
]
