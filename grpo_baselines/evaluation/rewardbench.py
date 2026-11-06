"""RewardBench OOD evaluation runner (inference is *not* executed by this project).

The runner is written so that the expensive part -- generation -- is a single pluggable
callable.  Everything else (prompt construction, both-ordering expansion, parsing, record
writing, aggregation) is deterministic, CPU-only code that the CPU test suite exercises
directly.  That split is deliberate: it lets the implementation be verified without a GPU,
while keeping a real, runnable inference path for whoever has one.

Two hard rules are enforced in code rather than in prose:

* **Every test sample is written.**  A row is emitted for each (qid, ordering) pair whether
  or not the completion parsed; unparsable rows carry ``parsed_ok=False`` and a
  ``parse_reason``.  Nothing is filtered before aggregation, and the aggregate functions
  report the invalid rate next to accuracy (see :mod:`evaluation.metrics`).
* **Both orderings are always produced.**  The official RewardBench score is computed on
  the stored ordering, but J1/J4R are trained on two states, so scoring them on one
  ordering only would evaluate a different prompt distribution than the one they trained
  on.  ``--views`` can restrict the run, and the report says which orderings exist.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from common import labels, parsing, paths, prompts

from . import metrics as metrics_mod

__all__ = [
    "PREDICTION_FIELDS",
    "GenerationFn",
    "build_prompt_messages",
    "build_prediction_record",
    "evaluate",
    "load_predictions",
    "main",
    "make_transformers_generator",
    "summarise_predictions",
    "write_summary",
]

#: Fields present on every per-sample prediction row.  Frozen so a downstream reader (and
#: the tests) can assert on the schema instead of guessing.
PREDICTION_FIELDS = (
    "qid",
    "view_id",
    "source_subset",
    "is_swapped",
    "display_order",
    "ground_truth",
    "question",
    "answer_a",
    "answer_b",
    "raw_completion",
    "parsed_ok",
    "parse_reason",
    "format_ok",
    "outcome",
    "chosen_response",
    "probability",
    "probability_target",
    "probability_decision",
    "correct",
    "latency_seconds",
    "method",
    "model_path",
)

#: A generator maps the rendered prompt (a list of chat messages) to completion text.
GenerationFn = Callable[[List[Dict[str, str]]], str]


def build_prompt_messages(method: str, view: Dict[str, Any]) -> List[Dict[str, str]]:
    """Render one evaluation view into chat messages for ``method``.

    The prompts come from :mod:`common.prompts`, the same module the trainers use, so the
    evaluation prompt cannot drift away from the training prompt.
    """
    return prompts.build_messages(
        method,
        question=view["question"],
        response_a=view["answer_a"],
        response_b=view["answer_b"],
    )


def build_prediction_record(
    *,
    method: str,
    view: Dict[str, Any],
    completion: str,
    latency_seconds: float,
    model_path: str = "",
) -> Dict[str, Any]:
    """Parse one completion into a complete, auditable prediction row.

    ``outcome`` is the display-space letter the judge chose.  ``chosen_response`` maps it
    back to the *physical* response it refers to (``answer1``/``answer2`` of the source
    pair), which is what position consistency compares across orderings -- comparing
    display letters across a swap would be meaningless because the correct letter flips.
    """
    parser = parsing.get_parser(method)
    parsed = parser(completion or "")
    outcome = parsed.outcome
    if outcome is None:
        chosen_response = None
    else:
        chosen_response = labels.display_to_response(outcome, bool(view["is_swapped"]))

    target = None
    if method == "jwc":
        target = labels.probability_target(view["ground_truth"])

    # The probability head never emits a letter, so everything downstream that speaks in
    # display space has to be derived from the decision rule the reward uses
    # (``rewards.jwc_brier``: p > 0.5 -> B, otherwise A).  Without this the JwC rows would
    # have ``correct=None`` and no physical choice, and the whole per-subset /
    # both-order / position-consistency report would refuse to aggregate them -- i.e. the
    # one method that *has* a calibrated head would be the one that cannot be scored.
    decision = None
    if method == "jwc" and parsed.ok and parsed.probability is not None:
        decision = (labels.ANSWER_B if parsed.probability > 0.5 else labels.ANSWER_A)
        if outcome is None:
            outcome = decision
            chosen_response = labels.display_to_response(
                outcome, bool(view["is_swapped"]))

    correct: Optional[bool] = None
    if parsed.ok and outcome is not None:
        correct = labels.is_correct(outcome, view["ground_truth"])

    record = {
        "qid": str(view["qid"]),
        "view_id": view["view_id"],
        "source_subset": view["source_subset"],
        "is_swapped": bool(view["is_swapped"]),
        "display_order": list(labels.display_order(bool(view["is_swapped"]))),
        "ground_truth": view["ground_truth"],
        "question": view["question"],
        "answer_a": view["answer_a"],
        "answer_b": view["answer_b"],
        "raw_completion": completion,
        "parsed_ok": bool(parsed.ok),
        "parse_reason": parsed.reason,
        "format_ok": bool(parsed.format_ok),
        "outcome": outcome,
        "chosen_response": chosen_response,
        "probability": parsed.probability,
        "probability_target": target,
        "probability_decision": decision,
        "correct": correct,
        "latency_seconds": float(latency_seconds),
        "method": method,
        "model_path": model_path,
    }
    missing = [field for field in PREDICTION_FIELDS if field not in record]
    if missing:  # pragma: no cover - guards future edits to this function
        raise AssertionError("prediction record is missing {0}".format(missing))
    return record


def evaluate(
    *,
    method: str,
    generate: GenerationFn,
    output_path: str,
    model_path: str = "",
    limit: Optional[int] = None,
    views: Sequence[str] = ("original", "swapped"),
    source_path: Optional[str] = None,
    progress_every: int = 100,
    resume: bool = True,
) -> Dict[str, Any]:
    """Run the evaluation and write one JSONL row per (sample, ordering).

    ``generate`` is the only place that touches a model.  It receives chat messages and
    returns the completion text, so a test can pass a stub and exercise the entire
    pipeline without a GPU.
    """
    from data import rewardbench_adapter

    prompts.check_method(method)
    requested = set(views)
    unknown = requested - {"original", "swapped"}
    if unknown:
        raise ValueError("unknown view names: {0}".format(sorted(unknown)))

    output_dir = os.path.dirname(os.path.abspath(output_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    source = os.path.realpath(source_path or str(paths.REWARDBENCH_OOD))
    model = os.path.realpath(model_path) if model_path else ""
    completed: Dict[str, Dict[str, Any]] = {}
    if resume and os.path.exists(output_path):
        previous = load_predictions(output_path)
        for record in previous:
            if (record["method"] != method or
                    (model and os.path.realpath(record["model_path"]) != model)):
                raise ValueError("resume predictions belong to a different method/model")
            view_id = str(record["view_id"])
            if view_id in completed:
                raise ValueError("duplicate view_id in resume predictions: " + view_id)
            completed[view_id] = record
        print("resuming from {0} valid completed views".format(len(completed)), flush=True)

    generated = 0
    qids = {record["qid"] for record in completed.values()}
    with open(output_path, "a" if completed else "w", encoding="utf-8") as handle:
        for view in rewardbench_adapter.iter_eval_views(source, limit=limit):
            name = "swapped" if view["is_swapped"] else "original"
            if name not in requested or str(view["view_id"]) in completed:
                continue
            messages = build_prompt_messages(method, view)
            started = time.time()
            completion = generate(messages)
            latency = time.time() - started
            record = build_prediction_record(
                method=method,
                view=view,
                completion=completion,
                latency_seconds=latency,
                model_path=model_path,
            )
            # Persist every result, including parse failures, before starting the next
            # inference.  A killed long run therefore never destroys completed views.
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            completed[str(record["view_id"])] = record
            generated += 1
            qids.add(record["qid"])
            if progress_every and generated % progress_every == 0:
                print("scored {0} new views; {1} total views ({2} qids)".format(
                    generated, len(completed), len(qids)), flush=True)

    expected = {
        view["view_id"]: view
        for view in rewardbench_adapter.iter_eval_views(source, limit=limit)
        if ("swapped" if view["is_swapped"] else "original") in requested
    }
    missing = sorted(set(expected) - set(completed))
    unexpected = sorted(set(completed) - set(expected))
    if missing or unexpected:
        raise ValueError("incomplete or mismatched evaluation: missing={0}, unexpected={1}".format(
            missing[:10], unexpected[:10]))
    # Verify the retained prediction is for exactly this fixed input pair, not merely
    # a matching qid.  This also guards resuming after an edited source/model prompt.
    for view_id, record in completed.items():
        view = expected[view_id]
        for field in ("qid", "source_subset", "is_swapped", "ground_truth",
                      "question", "answer_a", "answer_b"):
            if record[field] != view[field]:
                raise ValueError("resume prediction input mismatch: {0} field {1}".format(
                    view_id, field))
    ordered = [completed[view_id] for view_id in expected]
    summary = summarise_predictions(ordered, method=method)
    summary["output_path"] = os.path.abspath(output_path)
    summary["views_run"] = sorted(requested)
    summary["source"] = source
    with open(source, "rb") as source_handle:
        summary["source_sha256"] = hashlib.sha256(source_handle.read()).hexdigest()
    summary["resumed"] = len(completed) - generated > 0
    summary["new_views"] = generated
    write_summary(output_path, summary)
    write_summary_csv(output_path, summary)
    return summary


def write_summary_csv(prediction_path: str, summary: Dict[str, Any]) -> str:
    """Stable per-subset table; these are fixed OOD aggregates, not a new benchmark."""
    target = prediction_path + ".subsets.csv"
    fields = ["method", "subset", "pairs", "original_accuracy_all",
              "original_accuracy_parsed", "swapped_accuracy_all",
              "swapped_accuracy_parsed", "both_order_accuracy", "position_consistency"]
    with open(target, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for subset, item in summary["per_subset"].items():
            writer.writerow({
                "method": summary["method"], "subset": subset, "pairs": item["pairs"],
                "original_accuracy_all": item["original_accuracy_all_treated_wrong"],
                "original_accuracy_parsed": item["original_accuracy_on_parsed"],
                "swapped_accuracy_all": item.get("swapped_accuracy_all_treated_wrong"),
                "swapped_accuracy_parsed": item.get("swapped_accuracy_on_parsed"),
                "both_order_accuracy": item.get("both_order_accuracy"),
                "position_consistency": item.get("position_consistency"),
            })
    return target


def load_predictions(path: str) -> List[Dict[str, Any]]:
    """Read a predictions JSONL back, validating the schema before anything is aggregated."""
    records = []
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            missing = [field for field in PREDICTION_FIELDS if field not in record]
            if missing:
                raise ValueError(
                    "prediction row {0} in {1} is missing {2}".format(
                        line_number, path, missing))
            records.append(record)
    if not records:
        raise ValueError("no prediction rows in {0}".format(path))
    return records


def summarise_predictions(records: Sequence[Dict[str, Any]], *, method: str) -> Dict[str, Any]:
    """Aggregate predictions, keeping original and swapped orderings visible separately."""
    return metrics_mod.full_report(records, method=method)


def write_summary(prediction_path: str, summary: Dict[str, Any]) -> str:
    """Write ``<predictions>.summary.json`` next to the predictions file."""
    target = prediction_path + ".summary.json"
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
    return target


# --------------------------------------------------------------------------------------
# inference back ends (not executed by this project)
# --------------------------------------------------------------------------------------
def make_transformers_generator(
    model_path: str,
    *,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    batch_size: int = 1,
    dtype: str = "bfloat16",
) -> GenerationFn:
    """Build a greedy/sampled generator from a local ``transformers`` checkpoint.

    Deliberately kept behind a factory: importing :mod:`torch` and
    :mod:`transformers` at module scope would make every CPU test require a GPU stack.
    """
    del batch_size  # one completion at a time; batching would change nothing but memory
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not os.path.isdir(model_path):
        raise FileNotFoundError("model_path {0!r} is not a directory".format(model_path))

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=getattr(torch, dtype),
        device_map=os.environ.get("GRPO_EVAL_DEVICE", "auto") if torch.cuda.is_available() else None,
    )
    model.eval()

    def generate(messages: List[Dict[str, str]]) -> str:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0.0,
                temperature=temperature if temperature > 0.0 else None,
                top_p=1.0,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        completion = output[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(completion, skip_special_tokens=True)

    return generate


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Score a local judge checkpoint on RewardBench (both orderings).")
    parser.add_argument("--method", required=True, choices=list(prompts.VALID_METHODS))
    parser.add_argument("--model-path", required=True,
                        help="local checkpoint directory, e.g. runs/vanilla/checkpoint-final")
    parser.add_argument("--output", required=True, help="predictions JSONL to write")
    parser.add_argument("--source", default=None,
                        help="RewardBench JSONL (defaults to the fixed OOD file)")
    parser.add_argument("--views", default="original,swapped",
                        help="comma separated subset of {original,swapped}")
    parser.add_argument("--limit", type=int, default=None,
                        help="debug only: stop after N source rows (breaks comparability)")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args(argv)

    method = args.method
    generate = make_transformers_generator(
        args.model_path,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    views = tuple(part.strip() for part in args.views.split(",") if part.strip())
    summary = evaluate(
        method=method,
        generate=generate,
        output_path=args.output,
        model_path=args.model_path,
        limit=args.limit,
        views=views,
        source_path=args.source,
    )
    json.dump(summary, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
