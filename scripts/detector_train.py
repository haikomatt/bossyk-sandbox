#!/usr/bin/env python
"""Train ONE detector cell (phase-detector-training.md build-order step 4):
one (family, train_domain, seed). Loads the domain's v2 train/val split,
runs Amendment 3's calib-slice protocol + class-weighted fine-tuning via
`bossyk_sandbox.detector.train_backend.run_training_cell`, and appends one
row to a JSONL training manifest (brief step 7: family, train domain, seed,
slice seed, epochs, early-stopping epoch, class weights, temperature
fitted, wall time, cost estimate).

`--dry-run` uses `StubTrainBackend` (no torch, hermetic, CPU) -- the brief's
Phase A item 4 smoke mode; without it, `HFTrainBackend` is used (POD ONLY --
needs the `detector-train` extra: `uv sync --extra detector-train`).

Checkpoints are written under `--checkpoint-root` (default a local scratch
dir, NOT `probes/detector/results/`) -- model weights are never committed
(brief: "Download results (metrics, manifests, per-item scores; NOT model
weights)").

The single-cell logic (`build_backend`/`train_one_cell`/`append_manifest_row`)
lives in `bossyk_sandbox.detector.train_cli_core`, shared unchanged with
`scripts/detector_pod_train_all.py` (the all-18-cells pod orchestrator) --
this script is a thin CLI over that core, re-exporting its functions so
existing hermetic tests (`tests/unit/test_detector_train_script.py`) keep
working unchanged.

Usage:
    uv run python scripts/detector_train.py \\
        --family deberta --train-domain retail --seed 0 \\
        --checkpoint-root /workspace/checkpoints \\
        --manifest-out probes/detector/results/training_manifest.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bossyk_sandbox.detector.corpus_io import DEFAULT_DATA_ROOT
from bossyk_sandbox.detector.train_backend import FAMILIES
from bossyk_sandbox.detector.train_cli_core import (
    append_manifest_row,
    build_backend,
    train_one_cell,
)

__all__ = ["append_manifest_row", "build_backend", "train_one_cell", "main"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--family", required=True, choices=FAMILIES)
    parser.add_argument("--train-domain", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--slice-seed", type=int, default=None, help="defaults to --seed")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--corpus-version", default="v2")
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--manifest-out", type=Path, default=None)
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument(
        "--dry-run", action="store_true", help="StubTrainBackend, no torch, hermetic"
    )
    args = parser.parse_args(argv)

    slice_seed = args.slice_seed if args.slice_seed is not None else args.seed
    backend = build_backend(dry_run=args.dry_run)

    result = train_one_cell(
        family=args.family,
        train_domain=args.train_domain,
        seed=args.seed,
        slice_seed=slice_seed,
        data_root=args.data_root,
        corpus_version=args.corpus_version,
        checkpoint_root=args.checkpoint_root,
        backend=backend,
        max_epochs=args.max_epochs,
        early_stopping_patience=args.early_stopping_patience,
        per_device_batch_size=args.batch_size,
        learning_rate=args.lr,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )

    if args.manifest_out:
        append_manifest_row(args.manifest_out, result)
        print(f"appended manifest row to {args.manifest_out}", file=sys.stderr)

    print(
        f"trained {result.family}/{result.train_domain}/seed{result.seed}: "
        f"val_loss={result.final_val_loss:.4f} early_stop_epoch={result.early_stop_epoch} "
        f"T={result.calib_temperature:.3f} wall={result.wall_seconds:.1f}s "
        f"checkpoint={result.checkpoint_dir}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
