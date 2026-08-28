"""fast_auroc must agree with the canonical interp.correlate.auroc exactly
(float64 precision) -- it is a speed-only substitute used inside the
hierarchical bootstrap loop (step 5), never for a reported point estimate."""

from __future__ import annotations

import numpy as np

from bossyk_sandbox.detector.fast_auroc import fast_auroc
from bossyk_sandbox.interp.correlate import auroc


def test_fast_auroc_matches_canonical_no_ties() -> None:
    rng = np.random.default_rng(0)
    scores = rng.normal(size=200)
    y = rng.integers(0, 2, size=200).astype(bool)
    expected = auroc(scores.tolist(), y.tolist())
    got = fast_auroc(scores, y)
    assert abs(got - expected) < 1e-12


def test_fast_auroc_matches_canonical_heavy_ties() -> None:
    rng = np.random.default_rng(1)
    # Only 5 distinct score values over 300 items -> heavy tie blocks.
    scores = rng.integers(0, 5, size=300).astype(np.float64)
    y = rng.integers(0, 2, size=300).astype(bool)
    expected = auroc(scores.tolist(), y.tolist())
    got = fast_auroc(scores, y)
    assert abs(got - expected) < 1e-12


def test_fast_auroc_matches_canonical_all_identical_scores() -> None:
    scores = np.zeros(50)
    y = np.array([True] * 20 + [False] * 30)
    expected = auroc(scores.tolist(), y.tolist())
    got = fast_auroc(scores, y)
    assert abs(got - expected) < 1e-12
    assert got == 0.5  # fully tied scores -> chance


def test_fast_auroc_single_class_is_nan() -> None:
    scores = np.array([0.1, 0.5, 0.9])
    y = np.array([False, False, False])
    assert np.isnan(fast_auroc(scores, y))


def test_fast_auroc_length_mismatch_raises() -> None:
    try:
        fast_auroc(np.array([0.1, 0.2]), np.array([True]))
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def test_fast_auroc_matches_canonical_many_random_trials() -> None:
    rng = np.random.default_rng(42)
    for trial in range(20):
        n = rng.integers(10, 120)
        # Bias toward more ties on some trials.
        n_distinct = rng.integers(2, n + 1)
        scores = rng.integers(0, n_distinct, size=n).astype(np.float64)
        y = rng.integers(0, 2, size=n).astype(bool)
        if y.sum() == 0 or (~y).sum() == 0:
            continue
        expected = auroc(scores.tolist(), y.tolist())
        got = fast_auroc(scores, y)
        assert abs(got - expected) < 1e-9, (trial, expected, got)
