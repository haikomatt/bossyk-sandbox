#!/usr/bin/env python
"""Run-2 sizing calculator CLI (hermetic diversity fix, item 4:
phase-detector-training-step2-datagen.md Issues & Fixes / Part B run 1 fix
plan, step 3). Thin CLI over `bossyk_sandbox.interp.size_calc.size_run`.
Pure arithmetic -- no network, no LLM, no filesystem writes.

Given a per-domain MEASURED unique-rate and violation base rate (from a
small calibration sample) plus the REAL per-call cost, prints the call
volume and $ needed to reach each target positives-per-cell -- default 150
and 300, the dossier's pre-registered power-floor bounds
(a-fine-tuned-violation-detector-transfers-cross-domain.md). Feeds the run-2
spend-authorization ask; does not itself authorize or spend anything.

Usage:
    uv run python scripts/size_run.py --domain retail \\
        --unique-rate 0.017 --violation-rate 0.04 --cost-per-call-usd 0.002
"""

from __future__ import annotations

import argparse
import sys

from bossyk_sandbox.interp.size_calc import size_run


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--domain", required=True)
    parser.add_argument(
        "--unique-rate",
        type=float,
        required=True,
        help="measured unique decisions / calls (from a calibration sample)",
    )
    parser.add_argument(
        "--violation-rate",
        type=float,
        required=True,
        help="measured violations / unique decisions (from a calibration sample)",
    )
    parser.add_argument(
        "--cost-per-call-usd",
        type=float,
        required=True,
        help="the REAL per-call cost (from the provider dashboard, not a placeholder)",
    )
    parser.add_argument(
        "--targets",
        type=int,
        nargs="+",
        default=[150, 300],
        help="target positives-per-cell to size for (default: the dossier's 150/300 bounds)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    for target in args.targets:
        try:
            estimate = size_run(
                domain=args.domain,
                unique_rate=args.unique_rate,
                violation_rate=args.violation_rate,
                cost_per_call_usd=args.cost_per_call_usd,
                target_positives=target,
            )
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(
            f"domain={estimate.domain} target={estimate.target_positives} "
            f"unique_decisions_needed={estimate.unique_decisions_needed} "
            f"calls={estimate.calls_needed} "
            f"estimated_cost_usd={estimate.estimated_cost_usd:.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
