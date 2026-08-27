"""Shared core for `scripts/detector_train.py` (one cell) and
`scripts/detector_pod_train_all.py` (all 18 cells, on-pod orchestrator) --
extends rather than duplicates the single-cell training logic, per house
"extend before create" rule.
"""

from __future__ import annotations

import json
from pathlib import Path

from bossyk_sandbox.detector.corpus_io import load_domain_splits
from bossyk_sandbox.detector.pod_budget import A40_HOURLY_USD, cost_estimate_usd
from bossyk_sandbox.detector.train_backend import (
    FAMILIES,
    StubTrainBackend,
    TrainBackend,
    TrainConfig,
    TrainResult,
    run_training_cell,
)


def build_backend(*, dry_run: bool) -> TrainBackend:
    """`StubTrainBackend` for `--dry-run` (hermetic, CPU); otherwise the real
    `HFTrainBackend`, imported lazily so this module never requires torch."""
    if dry_run:
        return StubTrainBackend()
    from bossyk_sandbox.detector.train_backend import HFTrainBackend  # lazy: needs torch

    return HFTrainBackend()


def train_one_cell(
    *,
    family: str,
    train_domain: str,
    seed: int,
    slice_seed: int,
    data_root: Path,
    corpus_version: str,
    checkpoint_root: Path,
    backend: TrainBackend,
    max_epochs: int,
    early_stopping_patience: int,
    per_device_batch_size: int,
    learning_rate: float,
    lora_r: int,
    lora_alpha: int,
) -> TrainResult:
    if family not in FAMILIES:
        raise ValueError(f"unknown family: {family!r} (expected one of {FAMILIES})")
    splits = load_domain_splits(train_domain, data_root=data_root, corpus_version=corpus_version)
    output_dir = checkpoint_root / family / train_domain / f"seed{seed}"
    cfg = TrainConfig(
        family=family,
        train_domain=train_domain,
        seed=seed,
        slice_seed=slice_seed,
        train_rows=splits.train,
        val_rows=splits.val,
        output_dir=output_dir,
        max_epochs=max_epochs,
        early_stopping_patience=early_stopping_patience,
        per_device_batch_size=per_device_batch_size,
        learning_rate=learning_rate,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
    )
    return run_training_cell(cfg, backend)


def append_manifest_row(manifest_out: Path, result: TrainResult) -> None:
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    row = result.to_manifest_dict()
    row["cost_estimate_usd"] = cost_estimate_usd(result.wall_seconds, hourly_usd=A40_HOURLY_USD)
    with manifest_out.open("a") as f:
        f.write(json.dumps(row) + "\n")


def read_completed_cells(manifest_path: Path) -> set[tuple[str, str, int]]:
    """`(family, train_domain, seed)` keys already present in a manifest
    JSONL -- lets the all-cells orchestrator resume after an interruption
    (budget hit, transient failure, SSH drop) without re-training a cell
    that already has a row."""
    if not manifest_path.exists():
        return set()
    completed: set[tuple[str, str, int]] = set()
    for line in manifest_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        completed.add((row["family"], row["train_domain"], int(row["seed"])))
    return completed
