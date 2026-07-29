#!/usr/bin/env python
"""Populate the general-failure (is_error) label on a decisions dataset.

Reads a decisions JSON (from scripts/make_decisions.py, which now stores each
turn's `action`), runs the coherence judge over each item's (context, action),
and writes the dataset back with `is_error` set -- the non-lexical confound label
the activation re-probe needs (policy AUROC must beat the error AUROC, not just
the shuffled floor).

Runs as a SEPARATE step from make_decisions so the judge is re-runnable and
auditable without re-driving the (billable, slow) agent, and so a judge-model
change doesn't force a re-generation.

Gated on RUN_JUDGE_COHERENCE=1: it makes real, billable judge calls.

Usage:
    RUN_JUDGE_COHERENCE=1 FIREWORKS_API_KEY=... \\
        uv run python scripts/judge_coherence.py --in decisions.json \\
        --out decisions_judged.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from bossyk_sandbox.conditions.adversary_providers import ChatModel
from bossyk_sandbox.env import load_project_env
from bossyk_sandbox.interp.coherence_judge import build_fireworks_judge_llm, judge_items


def judge_rows(
    rows: list[dict[str, Any]],
    *,
    llm: ChatModel,
    max_attempts: int = 4,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict[str, int]:
    """Judge the coherence of every row that carries an `action`, writing
    `is_error` back into that row IN PLACE. Rows without an action, or whose
    verdict never resolved, are left untouched (no fabricated label). Returns
    counts {n_rows, n_judgeable, n_judged, n_error, n_unjudged}. Pure given a
    deterministic `llm` -- the IO (read/write JSON) stays in main."""
    judgeable = [(i, row) for i, row in enumerate(rows) if row.get("action") is not None]
    pairs = [(str(row["prompt"]), str(row["action"])) for _, row in judgeable]
    verdicts = judge_items(pairs, llm=llm, max_attempts=max_attempts, sleeper=sleeper)

    n_error = n_judged = 0
    for (row_idx, _row), verdict in zip(judgeable, verdicts, strict=True):
        if verdict is None:
            continue
        rows[row_idx]["is_error"] = verdict.is_error
        n_judged += 1
        n_error += int(verdict.is_error)
    return {
        "n_rows": len(rows),
        "n_judgeable": len(judgeable),
        "n_judged": n_judged,
        "n_error": n_error,
        "n_unjudged": len(judgeable) - n_judged,
    }


def main(argv: list[str] | None = None) -> int:
    load_project_env()
    parser = argparse.ArgumentParser(
        description="add is_error (coherence) labels to a decisions set"
    )
    parser.add_argument("--in", dest="in_path", required=True, help="input decisions JSON")
    parser.add_argument("--out", required=True, help="output decisions JSON (is_error populated)")
    parser.add_argument("--model", default=None, help="judge model id (default: Fireworks kimi)")
    parser.add_argument("--max-attempts", type=int, default=4)
    args = parser.parse_args(argv)

    if os.environ.get("RUN_JUDGE_COHERENCE") != "1":
        print("Set RUN_JUDGE_COHERENCE=1 to run the (billable) coherence judge.", file=sys.stderr)
        return 1

    rows: list[dict[str, Any]] = json.loads(Path(args.in_path).read_text())
    if not any(row.get("action") is not None for row in rows):
        print("no rows carry an 'action'; nothing to judge.", file=sys.stderr)
        return 1

    llm = (
        build_fireworks_judge_llm() if args.model is None else build_fireworks_judge_llm(args.model)
    )
    counts = judge_rows(rows, llm=llm, max_attempts=args.max_attempts)

    Path(args.out).write_text(json.dumps(rows, indent=2))
    print(
        f"wrote {args.out}: {counts['n_judged']}/{counts['n_rows']} judged "
        f"({counts['n_error']} error / {counts['n_judged'] - counts['n_error']} coherent), "
        f"{counts['n_unjudged']} unjudged"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
