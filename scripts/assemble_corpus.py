#!/usr/bin/env python
"""Part A item 2 CLI: assemble a generated decisions.jsonl into deduped,
stratified-group-split train/val/test files
(phase-detector-training-step2-datagen.md). Thin CLI over
`bossyk_sandbox.interp.corpus_assembly`. Pure/offline -- no network, no LLM.

Usage:
    uv run python scripts/assemble_corpus.py --domain retail
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bossyk_sandbox.interp.corpus_assembly import (
    dedupe,
    load_decisions_jsonl,
    stratified_group_split,
)

DATA_ROOT = Path(__file__).parent.parent / "probes" / "detector" / "data"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--decisions-path", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--near-dup-threshold", type=float, default=0.9)
    parser.add_argument("--shingle-size", type=int, default=5)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    decisions_path = args.decisions_path or (DATA_ROOT / args.domain / "decisions.jsonl")
    out_dir = args.out_dir or (DATA_ROOT / args.domain)

    records = load_decisions_jsonl(decisions_path)
    deduped, dedupe_report = dedupe(
        records, near_dup_threshold=args.near_dup_threshold, shingle_size=args.shingle_size
    )
    splits, split_report = stratified_group_split(
        deduped, val_frac=args.val_frac, test_frac=args.test_frac, seed=args.seed
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    for split_name, rows in splits.items():
        (out_dir / f"{split_name}.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n" if rows else ""
        )

    print(
        f"domain={args.domain} n_input={dedupe_report.n_input} "
        f"exact_dropped={dedupe_report.n_exact_dropped} "
        f"near_dropped={dedupe_report.n_near_dropped} n_output={dedupe_report.n_output} "
        f"group_fields={split_report.group_fields} n_groups={split_report.n_groups}"
    )
    for split_name, counts in split_report.counts.items():
        print(f"  {split_name}: n={counts['n']} violations={counts['violations']}")
    print(f"wrote splits to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
