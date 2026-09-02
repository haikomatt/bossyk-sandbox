"""Unit tests for bossyk_sandbox.detector.calibration (ECE)."""

from __future__ import annotations

import numpy as np

from bossyk_sandbox.detector.calibration import expected_calibration_error


def test_perfectly_calibrated_scores_give_zero_ece() -> None:
    # 10 items per bin-ish, score == empirical rate within each bin exactly.
    scores = np.array([0.1] * 10 + [0.9] * 10)
    y = np.array([False] * 9 + [True] * 1 + [True] * 9 + [False] * 1)
    ece = expected_calibration_error(scores, y, n_bins=10)
    assert ece == 0.0


def test_maximally_miscalibrated_scores_give_high_ece() -> None:
    # confident-violation predictions that are always wrong.
    scores = np.array([0.95] * 10)
    y = np.array([False] * 10)
    ece = expected_calibration_error(scores, y, n_bins=10)
    assert ece > 0.9


def test_nan_scores_are_dropped_not_propagated() -> None:
    scores = np.array([0.2, np.nan, 0.8])
    y = np.array([False, True, True])
    ece = expected_calibration_error(scores, y, n_bins=10)
    assert not np.isnan(ece)


def test_empty_input_returns_nan() -> None:
    scores = np.array([], dtype=float)
    y = np.array([], dtype=bool)
    assert np.isnan(expected_calibration_error(scores, y))


def test_boundary_scores_0_and_1_are_binned_without_error() -> None:
    scores = np.array([0.0, 1.0])
    y = np.array([False, True])
    ece = expected_calibration_error(scores, y, n_bins=5)
    assert ece == 0.0
