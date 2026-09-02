"""Amendment-3 calibration-slice protocol (`a-fine-tuned-violation-detector-
transfers-cross-domain.md`, adopted 2026-08-27).

The v2 retail val split drew 0 violations (stratified group split over only
10 scenario groups), making the originally registered "temperature scaling
fit on a train-domain validation split" degenerate for retail-trained
models. Fix, uniform across all three domains: a seeded, group-aware (by
scenario), stratified 15% calibration slice is held out from within the
TRAIN split only; the detector trains on the remaining ~85%; temperature
scaling is fit on the slice. The val split keeps its original early-stopping
role (loss-only, needs no positives).

This is a THIN WRAPPER, not a second group-split implementation: it reuses
`interp.corpus_assembly.stratified_group_split` (call with `test_frac=0.0`
so it partitions into exactly two named buckets) per the "extend before
create" rule -- the same group-aware, label-stratified, seeded bin-packing
algorithm the original train/val/test split used, applied one level down.

No split file changes: this only ever touches an in-memory TRAIN split
already loaded by the caller. `probes/detector/data/**/v2/*.jsonl` are never
read or written here.
"""

from __future__ import annotations

from dataclasses import dataclass

from bossyk_sandbox.interp.corpus_assembly import (
    Record,
    SplitReport,
    default_group_fields,
    stratified_group_split,
)

# Amendment 3: 15% of TRAIN, held out for calibration; detector trains on the
# remaining ~85%.
CALIB_FRAC = 0.15


@dataclass(frozen=True)
class CalibSliceResult:
    """`train` (~85%) is what the detector is fit on; `calib` (~15%) is what
    temperature scaling is fit on. `report` is the underlying group-split's
    report (achieved counts/violations per bucket), useful for the training
    manifest."""

    train: list[Record]
    calib: list[Record]
    report: SplitReport


def calibration_slice_split(
    train_rows: list[Record],
    *,
    seed: int,
    calib_frac: float = CALIB_FRAC,
    group_fields: tuple[str, ...] | None = None,
) -> CalibSliceResult:
    """Split a domain's TRAIN split into an 85% train-proper bucket and a
    `calib_frac` (default 15%) calibration slice, whole-scenario-group at a
    time, stratified by `is_violation`, seeded for reproducibility. The slice
    seed belongs in the training manifest (Amendment 3: "Slice seed recorded
    in the training manifest.").

    Group-exclusive by construction (inherited from `stratified_group_split`):
    no scenario/group ever contributes rows to both `train` and `calib`, so
    the calibration fit never leaks a scenario the model trained on."""
    fields = group_fields if group_fields is not None else default_group_fields(train_rows)
    splits, report = stratified_group_split(
        train_rows,
        val_frac=calib_frac,
        test_frac=0.0,
        seed=seed,
        group_fields=fields,
    )
    return CalibSliceResult(train=splits["train"], calib=splits["val"], report=report)
