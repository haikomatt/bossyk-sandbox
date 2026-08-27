"""Frozen BoW+logreg baseline (phase-detector-training.md build-order step 3):
fit per-domain on that domain's v2 TRAIN split ONLY, hyperparameters chosen by
CV within train, evaluated over the pre-registered 3x3 transfer matrix
(in-domain = own test split, OOD = the tested domain's FULL unique corpus --
`a-fine-tuned-violation-detector-transfers-cross-domain.md` Amendment 2's
corrected OOD definition). Both vectorisers (fixed-vocab, hashing) per the
dossier's confound 2. No temperature scaling -- raw probabilities.

Reuses the repo's existing numpy logistic-regression probe
(`interp.probe.train_probe` / `probe_scores` / `crossval_oof_scores`) rather
than introducing a second implementation or an sklearn dependency -- the same
probe fitting code the H2 text-baseline control and the activation probe use.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from bossyk_sandbox.detector.calibration import expected_calibration_error
from bossyk_sandbox.interp.correlate import auroc
from bossyk_sandbox.interp.probe import LinearProbe, crossval_oof_scores, probe_scores, train_probe

Array = NDArray[np.float64]

# A modest log-spaced grid: wide enough to matter, small enough that a 5-fold
# CV per domain per vectoriser stays fast and deterministic.
DEFAULT_L2_GRID: tuple[float, ...] = (0.3, 1.0, 3.0, 10.0, 30.0)

# The dossier's OOD power floor (Amendment 2biii): an OOD cell is evaluated
# only if the tested domain's full unique corpus has >= this many positives.
POWER_GATE = 150


def select_l2_by_cv(
    x_train: Array,
    y_train: NDArray[np.bool_],
    *,
    l2_grid: tuple[float, ...] = DEFAULT_L2_GRID,
    n_splits: int = 5,
    seed: int = 0,
) -> float:
    """Pick L2 via deterministic within-train CV only: for each candidate, a
    5-fold stratified out-of-fold AUROC on TRAIN ONLY (never touches
    val/test/OOD). The candidate with the highest OOF AUROC wins; nan CV
    scores (a fold too degenerate to fit) always lose; ties keep the first
    (smallest) candidate in `l2_grid`'s declared order."""
    best_l2 = l2_grid[0]
    best_score = float("-inf")
    y_train_list = y_train.tolist()
    for l2 in l2_grid:
        oof, ya = crossval_oof_scores(x_train, y_train_list, n_splits=n_splits, l2=l2, seed=seed)
        mask = ~np.isnan(oof)
        score = auroc(oof[mask].tolist(), ya[mask].tolist()) if mask.sum() else float("nan")
        if not np.isnan(score) and score > best_score:
            best_score = score
            best_l2 = l2
    return best_l2


def fit_domain_probe(
    x_train: Array,
    y_train: NDArray[np.bool_],
    *,
    l2_grid: tuple[float, ...] = DEFAULT_L2_GRID,
    seed: int = 0,
) -> tuple[LinearProbe, float]:
    """Select L2 via CV within train, then refit on the FULL train split with
    the winning L2. Returns `(probe, chosen_l2)` -- `chosen_l2` is recorded
    in the frozen results for reproducibility."""
    l2 = select_l2_by_cv(x_train, y_train, l2_grid=l2_grid, seed=seed)
    probe = train_probe(x_train, y_train.tolist(), l2=l2)
    return probe, l2


@dataclass(frozen=True)
class CellResult:
    """One cell of the transfer matrix. `auroc`/`ece` are raw (no temperature
    scaling). `underpowered` flags cells below the dossier's OOD power gate
    (Amendment 2biii) -- reported, not silently scored as if adequately
    powered."""

    auroc: float
    ece: float
    n: int
    n_pos: int
    underpowered: bool


def evaluate_cell(
    probe: LinearProbe, x: Array, y: NDArray[np.bool_], *, power_gate: int = POWER_GATE
) -> CellResult:
    """Score `probe` on `(x, y)` and report AUROC + ECE + counts. AUROC is
    `nan` when `y` is single-class (undefined, not a spurious 0.5 -- same
    convention as `interp.probe`)."""
    scores = probe_scores(probe, x)
    n = int(len(y))
    n_pos = int(y.sum())
    cell_auroc = auroc(scores.tolist(), y.tolist()) if 0 < n_pos < n else float("nan")
    return CellResult(
        auroc=cell_auroc,
        ece=expected_calibration_error(scores, y),
        n=n,
        n_pos=n_pos,
        underpowered=n_pos < power_gate,
    )


@dataclass(frozen=True)
class MajorityBaselineResult:
    """The trivial always-predict-the-train-majority-label baseline. AUROC is
    undefined for a constant score (the caller reports `nan`, not a fake
    0.5); accuracy is the only informative number a constant predictor has."""

    majority_label: bool
    accuracy: float
    n: int
    n_pos: int


def majority_class_baseline(
    y_train: NDArray[np.bool_], y_eval: NDArray[np.bool_]
) -> MajorityBaselineResult:
    """Fit the majority label on `y_train`, evaluate accuracy on `y_eval`."""
    majority = bool(y_train.mean() > 0.5) if len(y_train) else False
    n = int(len(y_eval))
    preds = np.full(n, majority, dtype=bool)
    acc = float((preds == y_eval).mean()) if n else float("nan")
    return MajorityBaselineResult(
        majority_label=majority, accuracy=acc, n=n, n_pos=int(y_eval.sum())
    )
