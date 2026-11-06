#!/usr/bin/env python
"""Audit prompt lengths against the real Qwen tokenizer and each method's real template.

Run from the repository root::

    python -m scripts.audit_lengths --limit 2000          # quick pass
    python -m scripts.audit_lengths                      # every training row

Writes ``runs/length_audit.json`` (machine-readable, one record per method) and prints a
table.  Exits non-zero if any method's ``max_prompt_length`` would force truncation --
the audit's job is to *report* the overflow, and the script's job is to make sure a
truncating config cannot be shipped quietly.

The measurement is tokenisation only: the tokenizer is loaded with
``local_files_only=True`` and no weights are touched, so this runs on a CPU-only box.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List

# Allow `python scripts/audit_lengths.py` as well as `python -m scripts.audit_lengths`:
# in the first form sys.path[0] is scripts/, so the project root has to be added before
# the package imports below.
_PROJECT_ROOT = __file__.rsplit("/scripts/", 1)[0]
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from common import config as config_mod
from common import paths
from data import helpsteer_adapter, length_audit


def _records_for(method: str, *, limit: int | None) -> List[Dict[str, Any]]:
    """Training rows in the shape this method's prompt builder expects."""
    two_states = method in config_mod.TWO_STATE_METHODS
    return list(
        helpsteer_adapter.iter_training_rows(two_states=two_states, limit=limit)
    )


def audit_method(method: str, config: config_mod.Config, *, limit: int | None,
                 tokenizer: Any) -> Dict[str, Any]:
    records = _records_for(method, limit=limit)
    audit = length_audit.run_audit(
        tokenizer,
        method,
        records,
        config.max_prompt_length,
        config.max_completion_length,
    )
    audit["method"] = method
    audit["max_prompt_length"] = config.max_prompt_length
    audit["max_completion_length"] = config.max_completion_length
    return audit


def format_row(audit: Dict[str, Any]) -> str:
    tokens = audit["prompt_tokens"]
    overflow = audit["overflow_at_max_prompt_length"]
    return (
        "{method:<8} rows={rows:>6}  min={minn:>5}  p50={p50:>6}  p90={p90:>6}  "
        "p95={p95:>6}  p99={p99:>6}  max={maxx:>6}  >limit={over:>4} ({ratio:.2%})  "
        "budget={budget}"
    ).format(
        method=audit["method"],
        rows=audit["rows_measured"],
        minn=tokens["min"],
        p50=tokens["p50"],
        p90=tokens["p90"],
        p95=tokens["p95"],
        p99=tokens["p99"],
        maxx=tokens["max"],
        over=overflow["exceeding"],
        ratio=overflow["exceeding_ratio"],
        budget=audit["context_budget"],
    )


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="measure only the first N source pairs (default: every row)",
    )
    parser.add_argument(
        "--methods",
        default="all",
        help="comma-separated method names, or 'all' (default)",
    )
    parser.add_argument(
        "--output",
        default=str(paths.PROJECT_ROOT / "runs" / "length_audit.json"),
        help="where to write the JSON record",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit non-zero if any method overflows, even when only --limit rows were measured",
    )
    args = parser.parse_args(argv)

    if args.methods == "all":
        methods = list(config_mod.load_all_configs().keys())
    else:
        methods = [item.strip() for item in args.methods.split(",") if item.strip()]

    configs = config_mod.load_all_configs()
    for method in methods:
        if method not in configs:
            print("unknown method {0!r}; known: {1}".format(method, sorted(configs)),
                  file=sys.stderr)
            return 2

    tokenizer = length_audit.load_tokenizer()
    audits = {}
    for method in methods:
        audit = audit_method(method, configs[method], limit=args.limit, tokenizer=tokenizer)
        audits[method] = audit
        print(format_row(audit))

    payload = {
        "tokenizer": str(paths.BASE_MODEL),
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_vocab_size": len(tokenizer),
        "limit": args.limit,
        "context_limit": paths.MAX_CONTEXT_TOKENS,
        "methods": audits,
    }
    length_audit.write_audit(args.output, payload)
    print("\nwrote {0}".format(args.output))

    overflowing = [
        method for method, audit in audits.items()
        if audit["overflow_at_max_prompt_length"]["exceeding"]
    ]
    if overflowing:
        for method in overflowing:
            length_audit.assert_within_limits(audits[method])
        if args.strict:
            return 1
    if args.limit is None:
        for method in methods:
            length_audit.assert_within_limits(audits[method])
        print("all methods fit within max_prompt_length")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
