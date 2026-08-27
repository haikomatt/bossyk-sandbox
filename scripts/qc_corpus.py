#!/usr/bin/env python
"""Part A item 3 CLI: quality-check an assembled corpus
(phase-detector-training-step2-datagen.md). Thin CLI over
`bossyk_sandbox.interp.corpus_qc` + `corpus_assembly`. Pure/offline -- no
network, no LLM.

Runs a cross-split AND cross-domain leakage scan, a per-domain/per-split
class-balance report, exports a 50-item hand-audit sample per domain, and
writes a provenance/honesty manifest per domain. Exits non-zero if the
leakage scan finds anything.

Usage:
    uv run python scripts/qc_corpus.py --domains retail airline advice-eligibility
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from bossyk_sandbox.interp.corpus_assembly import (
    DEFAULT_CORPUS_VERSION,
    MixedCorpusVersionError,
    Record,
    assert_single_corpus_version,
    corpus_data_dir,
    dedupe_near_duplicates,
    default_group_fields,
    load_decisions_jsonl,
)
from bossyk_sandbox.interp.corpus_qc import (
    build_manifest,
    class_balance_report,
    cross_domain_near_duplicate_scan,
    export_hand_audit_sample,
    leakage_scan,
)

DATA_ROOT = Path(__file__).parent.parent / "probes" / "detector" / "data"
SPLIT_NAMES = ("train", "val", "test")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domains", nargs="+", default=["retail", "airline", "advice-eligibility"])
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--audit-n", type=int, default=50)
    parser.add_argument("--audit-seed", type=int, default=0)
    parser.add_argument("--near-dup-threshold", type=float, default=0.85)
    parser.add_argument("--shingle-size", type=int, default=5)
    parser.add_argument("--generator-model", default="unknown")
    parser.add_argument("--serving-path", default="unknown")
    parser.add_argument(
        "--split-seed", type=int, default=0, help="the seed assemble_corpus.py used"
    )
    parser.add_argument(
        "--cap-hit", default=None, help="propagate the generation run's cap_hit, if any"
    )
    parser.add_argument(
        "--corpus-version",
        default=DEFAULT_CORPUS_VERSION,
        help=(
            "corpus version this run is QC'ing (hermetic diversity fix, item 3). "
            "Also picks each domain's data subdirectory; "
            f"{DEFAULT_CORPUS_VERSION!r} is run-1's implicit, flat-path layout"
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="the generation run's sampling temperature, recorded in the manifest",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="the generation run's top_p (if any), recorded in the manifest",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    domain_dirs = {
        domain: corpus_data_dir(args.data_root, domain, args.corpus_version)
        for domain in args.domains
    }

    splits_by_domain: dict[str, dict[str, list[Record]]] = {}
    all_records: list[Record] = []
    for domain in args.domains:
        domain_dir = domain_dirs[domain]
        splits = {name: load_decisions_jsonl(domain_dir / f"{name}.jsonl") for name in SPLIT_NAMES}
        splits_by_domain[domain] = splits
        domain_records = [row for rows in splits.values() for row in rows]
        try:
            detected_version = assert_single_corpus_version(domain_records)
        except MixedCorpusVersionError as exc:
            print(f"error: domain {domain!r}: {exc}", file=sys.stderr)
            return 1
        if domain_records and detected_version != args.corpus_version:
            print(
                f"error: domain {domain!r} at {domain_dir} is corpus_version="
                f"{detected_version!r} but --corpus-version={args.corpus_version!r} was "
                "requested; refusing to silently mix corpus versions",
                file=sys.stderr,
            )
            return 1
        all_records.extend(domain_records)

    combined_splits: dict[str, list[Record]] = {name: [] for name in SPLIT_NAMES}
    for splits in splits_by_domain.values():
        for name, rows in splits.items():
            combined_splits[name].extend(rows)

    group_fields = default_group_fields(all_records) if all_records else ("scenario_id",)
    leak_report = leakage_scan(combined_splits, group_fields=group_fields)
    cross_domain_report = cross_domain_near_duplicate_scan(
        all_records, threshold=args.near_dup_threshold, shingle_size=args.shingle_size
    )
    balance = class_balance_report(combined_splits)

    print(
        f"leakage scan (group_fields={group_fields}): {'CLEAN' if leak_report.clean else 'LEAKED'}"
    )
    for leaked in leak_report.leaked_groups:
        print(f"  LEAK: {leaked.key} in splits {leaked.splits}")
    print(
        f"cross-domain near-dup scan: {len(cross_domain_report.pairs)}/"
        f"{cross_domain_report.n_pairs_checked} pairs (rate={cross_domain_report.rate:.4f})"
    )

    generated_at = datetime.now(UTC).isoformat()
    for domain in args.domains:
        domain_records = [r for r in all_records if r["domain"] == domain]
        audit_sample = export_hand_audit_sample(
            domain_records, n=args.audit_n, seed=args.audit_seed
        )
        # "audit" is a pseudo-domain (not a real per-domain data directory),
        # so it isn't routed through `corpus_data_dir` -- but it gets the
        # same v1-flat / versioned-subdirectory treatment for the same
        # reason: a run-2 audit sample must never silently overwrite run-1's.
        audit_dir = (
            args.data_root / "audit"
            if args.corpus_version == DEFAULT_CORPUS_VERSION
            else args.data_root / "audit" / args.corpus_version
        )
        audit_dir.mkdir(parents=True, exist_ok=True)
        audit_path = audit_dir / f"{domain}_sample50.jsonl"
        audit_path.write_text(
            "\n".join(json.dumps(r) for r in audit_sample) + ("\n" if audit_sample else "")
        )

        _, n_near_dup = dedupe_near_duplicates(
            domain_records, threshold=args.near_dup_threshold, shingle_size=args.shingle_size
        )
        near_dup_rate = (n_near_dup / len(domain_records)) if domain_records else 0.0

        domain_balance = balance.get(domain, {})
        counts = {name: int(domain_balance.get(name, {}).get("n", 0)) for name in SPLIT_NAMES}
        class_balance = {
            name: float(domain_balance.get(name, {}).get("violation_rate", 0.0))
            for name in SPLIT_NAMES
        }
        manifest = build_manifest(
            domain=domain,
            generator_model=args.generator_model,
            serving_path=args.serving_path,
            seed=args.split_seed,
            counts=counts,
            class_balance=class_balance,
            cap_hit=args.cap_hit,
            near_dup_rate_within_domain=near_dup_rate,
            generated_at=generated_at,
            corpus_version=args.corpus_version,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        manifest_path = domain_dirs[domain] / "manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2))

        rounded_balance = {k: round(v, 3) for k, v in class_balance.items()}
        print(
            f"{domain}: counts={counts} class_balance={rounded_balance} "
            f"near_dup_rate={near_dup_rate:.4f} audit_n={len(audit_sample)}"
        )
        print(f"  wrote {audit_path}")
        print(f"  wrote {manifest_path}")

    return 0 if leak_report.clean else 1


if __name__ == "__main__":
    raise SystemExit(main())
