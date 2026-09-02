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
import sys
from pathlib import Path

from bossyk_sandbox.interp.corpus_assembly import (
    DEFAULT_CORPUS_VERSION,
    MixedCorpusVersionError,
    assert_single_corpus_version,
    corpus_data_dir,
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
    parser.add_argument(
        "--corpus-version",
        default=DEFAULT_CORPUS_VERSION,
        help=(
            "corpus version this run is assembling (hermetic diversity fix, item 3). "
            "Also picks the default decisions-path/out-dir subdirectory; "
            f"{DEFAULT_CORPUS_VERSION!r} is run-1's implicit, flat-path layout"
        ),
    )
    return parser


def _default_paths(data_root: Path, domain: str, corpus_version: str) -> tuple[Path, Path]:
    domain_dir = corpus_data_dir(data_root, domain, corpus_version)
    return domain_dir / "decisions.jsonl", domain_dir


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    default_decisions_path, default_out_dir = _default_paths(
        DATA_ROOT, args.domain, args.corpus_version
    )
    decisions_path = args.decisions_path or default_decisions_path
    out_dir = args.out_dir or default_out_dir

    records = load_decisions_jsonl(decisions_path)
    try:
        detected_version = assert_single_corpus_version(records)
    except MixedCorpusVersionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if records and detected_version != args.corpus_version:
        print(
            f"error: {decisions_path} is corpus_version={detected_version!r} but "
            f"--corpus-version={args.corpus_version!r} was requested; refusing to "
            "silently mix corpus versions",
            file=sys.stderr,
        )
        return 1

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
