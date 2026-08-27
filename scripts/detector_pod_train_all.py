#!/usr/bin/env python
"""On-pod orchestrator: run ALL training + scoring cells for the pre-registered
sweep -- 3 train domains x 2 families x 3 seeds = 18 cells
(phase-detector-training.md build-order step 4) -- then score each trained
checkpoint over the 3x3 transfer matrix (step 4 item 2), sequentially, on
one pod.

Runs ON the pod (needs `detector-train` extra unless `--dry-run`). The LOCAL
supervisor (`scripts/detector_pod_runner.py`) launches this inside a tmux
session and is the thing that actually holds the RunPod API key / terminates
the pod -- this script's own responsibility is: don't start a cell it can't
finish inside the budget, and leave a resumable manifest if it has to stop
early.

Safety rails carried IN THIS SCRIPT (brief step 3):
- **Wall-clock budget**: `bossyk_sandbox.detector.pod_budget.should_terminate`
  is checked before every cell. If the budget is already exhausted, or the
  estimated time for one more cell (from a running average of completed
  cells, seeded with `--first-cell-estimate-seconds`) would exceed what's
  left, the loop stops WITHOUT starting that cell -- reported as
  `stopped_reason` in the run summary, not silently.
- **Checkpointing/resume**: cells already present in `--manifest-out` are
  skipped (`train_cli_core.read_completed_cells`) -- rerunning this script
  after an interruption picks up where it left off. HF `Trainer`'s own
  `checkpoint-*` dirs (`train_backend._find_resumable_checkpoint`) resume a
  cell that was interrupted mid-training.
- This script does NOT terminate the pod itself -- see
  `scripts/detector_pod_runner.py` for the `finally`/trap pod-lifecycle
  teardown; this script only decides when to STOP STARTING NEW WORK.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path
from typing import Any

from bossyk_sandbox.detector.corpus_io import DEFAULT_DATA_ROOT, DEFAULT_DOMAINS
from bossyk_sandbox.detector.pod_budget import (
    HARD_CAP_USD,
    MAX_WALL_SECONDS,
    remaining_seconds,
    should_terminate,
)
from bossyk_sandbox.detector.score_cli_core import score_cell
from bossyk_sandbox.detector.train_backend import FAMILIES, SEEDS
from bossyk_sandbox.detector.train_cli_core import (
    append_manifest_row,
    build_backend,
    read_completed_cells,
    train_one_cell,
)

# Conservative prior for "how long does one cell take" before any cell has
# completed on this run -- deliberately generous (Qwen2.5-0.5B LoRA / DeBERTa
# full-FT over a few thousand rows, a handful of epochs, on an A40) so the
# FIRST cell is never started when there plausibly isn't time to finish it.
DEFAULT_FIRST_CELL_ESTIMATE_SECONDS = 20 * 60


def run_all(
    *,
    families: tuple[str, ...],
    domains: tuple[str, ...],
    seeds: tuple[int, ...],
    data_root: Path,
    corpus_version: str,
    checkpoint_root: Path,
    manifest_out: Path,
    scores_root: Path,
    max_epochs: int,
    early_stopping_patience: int,
    per_device_batch_size: int,
    learning_rate: float,
    lora_r: int,
    lora_alpha: int,
    dry_run: bool,
    hard_cap_usd: float = HARD_CAP_USD,
    max_wall_seconds: int = MAX_WALL_SECONDS,
    first_cell_estimate_seconds: float = DEFAULT_FIRST_CELL_ESTIMATE_SECONDS,
    score_domains: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Train + score every (family, domain, seed) cell not already in
    `manifest_out`, stopping early (never mid-cell) if the budget wouldn't
    cover another one. Returns a run summary dict (also useful as the
    lifecycle-log payload for the local supervisor)."""
    run_start = time.monotonic()
    backend = build_backend(dry_run=dry_run)
    completed = read_completed_cells(manifest_out)
    all_cells = list(itertools.product(families, domains, seeds))
    remaining_cells = [c for c in all_cells if c not in completed]

    cell_durations: list[float] = []
    ran: list[tuple[str, str, int]] = []
    skipped_already_done = [c for c in all_cells if c in completed]
    stopped_reason: str | None = None

    for family, train_domain, seed in remaining_cells:
        elapsed = time.monotonic() - run_start
        decision = should_terminate(
            elapsed, hard_cap_usd=hard_cap_usd, max_wall_seconds=max_wall_seconds
        )
        if decision.terminate:
            stopped_reason = (
                f"budget exhausted before starting {family}/{train_domain}/seed{seed}: "
                f"{decision.reason}"
            )
            break

        left = remaining_seconds(
            elapsed, hard_cap_usd=hard_cap_usd, max_wall_seconds=max_wall_seconds
        )
        estimate = (
            sum(cell_durations) / len(cell_durations)
            if cell_durations
            else first_cell_estimate_seconds
        )
        if left < estimate:
            stopped_reason = (
                f"insufficient budget for another cell before "
                f"{family}/{train_domain}/seed{seed}: {left:.0f}s left, "
                f"~{estimate:.0f}s/cell estimated"
            )
            break

        cell_start = time.monotonic()
        result = train_one_cell(
            family=family,
            train_domain=train_domain,
            seed=seed,
            slice_seed=seed,
            data_root=data_root,
            corpus_version=corpus_version,
            checkpoint_root=checkpoint_root,
            backend=backend,
            max_epochs=max_epochs,
            early_stopping_patience=early_stopping_patience,
            per_device_batch_size=per_device_batch_size,
            learning_rate=learning_rate,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
        )
        score_cell(
            family=family,
            train_domain=train_domain,
            seed=seed,
            checkpoint_dir=result.checkpoint_dir,
            temperature=result.calib_temperature,
            data_root=data_root,
            corpus_version=corpus_version,
            domains=score_domains or domains,
            out_root=scores_root,
            backend=backend,
        )
        append_manifest_row(manifest_out, result)
        ran.append((family, train_domain, seed))
        cell_durations.append(time.monotonic() - cell_start)
        print(
            f"[{len(ran)}/{len(remaining_cells)}] done {family}/{train_domain}/seed{seed} "
            f"in {cell_durations[-1]:.0f}s",
            file=sys.stderr,
        )

    total_elapsed = time.monotonic() - run_start
    return {
        "requested_cells": len(all_cells),
        "already_done_at_start": [list(c) for c in skipped_already_done],
        "ran_this_invocation": [list(c) for c in ran],
        "remaining_after_this_invocation": [
            list(c) for c in all_cells if c not in completed and c not in ran
        ],
        "stopped_reason": stopped_reason,
        "total_wall_seconds": total_elapsed,
        "all_cells_complete": len(completed) + len(ran) == len(all_cells),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--families", default=",".join(FAMILIES))
    parser.add_argument("--domains", default=",".join(DEFAULT_DOMAINS))
    parser.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--corpus-version", default="v2")
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--manifest-out", type=Path, required=True)
    parser.add_argument("--scores-root", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, default=None)
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--hard-cap-usd", type=float, default=HARD_CAP_USD)
    parser.add_argument("--max-wall-seconds", type=int, default=MAX_WALL_SECONDS)
    parser.add_argument(
        "--first-cell-estimate-seconds", type=float, default=DEFAULT_FIRST_CELL_ESTIMATE_SECONDS
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    summary = run_all(
        families=tuple(f.strip() for f in args.families.split(",") if f.strip()),
        domains=tuple(d.strip() for d in args.domains.split(",") if d.strip()),
        seeds=tuple(int(s.strip()) for s in args.seeds.split(",") if s.strip()),
        data_root=args.data_root,
        corpus_version=args.corpus_version,
        checkpoint_root=args.checkpoint_root,
        manifest_out=args.manifest_out,
        scores_root=args.scores_root,
        max_epochs=args.max_epochs,
        early_stopping_patience=args.early_stopping_patience,
        per_device_batch_size=args.batch_size,
        learning_rate=args.lr,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        dry_run=args.dry_run,
        hard_cap_usd=args.hard_cap_usd,
        max_wall_seconds=args.max_wall_seconds,
        first_cell_estimate_seconds=args.first_cell_estimate_seconds,
    )

    if args.summary_out:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2), file=sys.stderr)
    return 0 if summary["all_cells_complete"] or summary["ran_this_invocation"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
