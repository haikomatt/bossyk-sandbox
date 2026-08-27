#!/usr/bin/env python
"""Score a trained detector checkpoint over the pre-registered 3x3 transfer
matrix (phase-detector-training.md build-order step 4, item 2): given ONE
trained (family, train_domain, seed) checkpoint, emit PER-ITEM scores for
every eval domain -- in-domain = that domain's v2 test split, OOD = the
tested domain's full unique corpus (Amendment 2's corrected OOD definition)
-- to `probes/detector/results/scores/<family>/<train_domain>/seed<k>/
<eval_domain>.jsonl`.

Both raw (T=1, i.e. plain sigmoid of the logit) and calibrated
(Amendment-3 temperature, fit on that run's calibration slice, passed in via
`--temperature` from the training manifest) scores are saved per item. This
script emits SCORES ONLY -- no paired-gap CIs, no verdict language; step 5
(verdict) is a separate, later, reviewed step.

`--dry-run` uses `StubTrainBackend` (hermetic, CPU); without it,
`HFTrainBackend` loads the real checkpoint (POD ONLY).

The scoring logic (`score_cell`) lives in
`bossyk_sandbox.detector.score_cli_core`, shared unchanged with
`scripts/detector_pod_train_all.py` (the all-18-cells pod orchestrator) --
this script is a thin CLI over that core.

Usage:
    uv run python scripts/detector_score.py \\
        --family deberta --train-domain retail --seed 0 --temperature 1.23 \\
        --checkpoint-root /workspace/checkpoints \\
        --out-root probes/detector/results/scores
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bossyk_sandbox.detector.corpus_io import DEFAULT_DATA_ROOT, DEFAULT_DOMAINS
from bossyk_sandbox.detector.score_cli_core import score_cell
from bossyk_sandbox.detector.train_backend import FAMILIES, StubTrainBackend, TrainBackend

__all__ = ["build_backend", "score_cell", "main"]


def build_backend(*, dry_run: bool) -> TrainBackend:
    if dry_run:
        return StubTrainBackend()
    from bossyk_sandbox.detector.train_backend import HFTrainBackend  # lazy: needs torch

    return HFTrainBackend()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--family", required=True, choices=FAMILIES)
    parser.add_argument("--train-domain", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--checkpoint-root", type=Path, default=None)
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=None, help="overrides --checkpoint-root"
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--corpus-version", default="v2")
    parser.add_argument("--domains", default=",".join(DEFAULT_DOMAINS))
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if args.checkpoint_dir is not None:
        checkpoint_dir = args.checkpoint_dir
    elif args.checkpoint_root is not None:
        checkpoint_dir = (
            args.checkpoint_root / args.family / args.train_domain / f"seed{args.seed}" / "final"
        )
    else:
        parser.error("one of --checkpoint-dir / --checkpoint-root is required")
        return 2  # unreachable, appeases mypy

    domains = tuple(d.strip() for d in args.domains.split(",") if d.strip())
    backend = build_backend(dry_run=args.dry_run)

    written = score_cell(
        family=args.family,
        train_domain=args.train_domain,
        seed=args.seed,
        checkpoint_dir=checkpoint_dir,
        temperature=args.temperature,
        data_root=args.data_root,
        corpus_version=args.corpus_version,
        domains=domains,
        out_root=args.out_root,
        backend=backend,
    )
    for eval_domain, path in written.items():
        print(f"wrote {eval_domain}: {path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
