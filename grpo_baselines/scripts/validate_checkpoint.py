"""Reload a finished checkpoint without allocating a third training GPU."""
import argparse
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    started = time.time()
    state_path = checkpoint / "trainer_state.json"
    if not state_path.is_file():
        raise FileNotFoundError(f"missing {state_path}")
    state = json.loads(state_path.read_text())
    if int(state.get("global_step", 0)) < 1:
        raise ValueError("checkpoint has no completed optimizer step")
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    config = AutoConfig.from_pretrained(checkpoint, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, local_files_only=True, dtype=torch.bfloat16,
    )
    parameter_count = sum(p.numel() for p in model.parameters())
    if parameter_count < 1:
        raise ValueError("reloaded checkpoint has no parameters")
    del model
    result = {
        "checkpoint": str(checkpoint),
        "reload_passed": True,
        "reload_device": "cpu",
        "model_type": config.model_type,
        "parameter_count": parameter_count,
        "tokenizer_class": type(tokenizer).__name__,
        "global_step": state["global_step"],
        "model_only_not_training_resumable": not (checkpoint / "optimizer.pt").is_file(),
        "elapsed_seconds": time.time() - started,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
