#!/usr/bin/env python
"""Regenerates the cross-domain H1 summary (code-review finding 17: no
script reproduced `docs/bench_output/phase2b_crossdomain_h1.json`).

Merges one or more per-domain H1 JSON files (as produced by
`scripts/h1_bench.py`) into the cross-domain shape: each domain's
canonical-strength `h1_overall`/`h1_by_class`/`h1_by_boundary` blocks plus a
`strength_sweep` of every swept strength's overall rate. Pure and
deterministic -- no network, no judge calls; the merge logic itself lives in
`bossyk_sandbox.scoring.crossdomain_merge.merge_crossdomain_h1` (unit-tested
in tests/unit/test_crossdomain_merge.py, including an acceptance check that
merging the two committed per-domain inputs reproduces the committed
cross-domain file).

Fails loudly (non-zero exit, no output written) if the inputs disagree on
adversary_model/guardrail_backend/intensity/canonical_strength, or if two
inputs claim the same domain -- comparing rates across domains attacked
under different configs would not be a meaningful cross-domain read.

Usage:
    uv run python scripts/h1_crossdomain_merge.py \\
        docs/bench_output/phase2a_h1.json \\
        docs/bench_output/phase2b_h1_retail.json \\
        --output docs/bench_output/phase2b_crossdomain_h1.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from bossyk_sandbox.scoring.crossdomain_merge import CrossDomainMergeError, merge_crossdomain_h1


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else None)
    parser.add_argument(
        "inputs",
        nargs="+",
        metavar="PER_DOMAIN_JSON",
        help="Per-domain H1 JSON files to merge (one per domain).",
    )
    parser.add_argument(
        "--output",
        required=True,
        metavar="PATH",
        help="Where to write the merged cross-domain H1 JSON.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    try:
        merged = merge_crossdomain_h1(args.inputs)
    except CrossDomainMergeError as exc:
        print(f"h1_crossdomain_merge: {exc}", file=sys.stderr)
        return 1

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(merged, indent=2))

    print(str(output_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
