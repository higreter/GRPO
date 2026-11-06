#!/usr/bin/env python3
"""Create an explicitly labelled one-step GPU-smoke config from a formal config.

This tool never edits the formal YAML.  It preserves the model, data source, templates,
reward, KL coefficient, rollout group size, sequence limits and optimizer type; it changes
only batch scheduling and the number of optimizer steps so a real smoke test can exercise
one full update in an isolated run directory.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common import config as config_mod
from common import paths


def build_smoke_data(formal: dict, run_output: str) -> tuple[dict, dict]:
    """Return one-step config data and a record of every intentional difference."""
    data = copy.deepcopy(formal)
    original = {
        key: data[key]
        for key in ("output_dir", "global_batch_size", "micro_batch_size",
                    "gradient_accumulation", "training_steps", "total_rollouts",
                    "estimated_generated_tokens")
    }
    states = int(data["states_per_sample"])
    generations = int(data["num_generations"])
    data["output_dir"] = run_output
    data["global_batch_size"] = 1
    data["micro_batch_size"] = 1
    data["gradient_accumulation"] = states
    data["training_steps"] = 1
    data["total_rollouts"] = generations
    data["estimated_generated_tokens"] = generations * (
        int(data["estimated_prompt_tokens"]) + int(data["max_completion_length"])
    )
    data["checkpoint"]["save_every_n_steps"] = 1
    data["checkpoint"]["save_total_limit"] = 1
    validated = config_mod.validate_config(data, source="generated GPU smoke config")
    changes = {
        "purpose": "one real optimizer step; never a formal result",
        "preserved": {
            "base_model": data["base_model"],
            "train_dataset": data["train_dataset"],
            "method": data["method"],
            "num_generations": data["num_generations"],
            "rollouts_per_state": validated.rollouts_per_state,
            "max_prompt_length": data["max_prompt_length"],
            "max_completion_length": data["max_completion_length"],
            "kl_beta": data["kl_beta"],
            "reference_model_policy": data["reference_model_policy"],
            "learning_rate": data["learning_rate"],
        },
        "overrides": {
            key: {"formal": original[key], "smoke": data[key]}
            for key in original
            if original[key] != data[key]
        },
    }
    return data, changes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-config", required=True)
    parser.add_argument("--run-output", required=True)
    parser.add_argument("--output-config", required=True)
    parser.add_argument("--output-metadata", required=True)
    args = parser.parse_args()

    with open(args.formal_config, "r", encoding="utf-8") as handle:
        formal = yaml.safe_load(handle)
    data, metadata = build_smoke_data(formal, args.run_output)
    config_path = paths.assert_writable_file(Path(args.output_config))
    metadata_path = paths.assert_writable_file(Path(args.output_metadata))
    with open(config_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, allow_unicode=True, sort_keys=False)
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"config": str(config_path), "metadata": str(metadata_path), **metadata}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
