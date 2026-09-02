"""Unit tests for bossyk_sandbox.detector.bow_baseline."""

from __future__ import annotations

import numpy as np

from bossyk_sandbox.detector.bow_baseline import (
    evaluate_cell,
    fit_domain_probe,
    majority_class_baseline,
    select_l2_by_cv,
)
from bossyk_sandbox.detector.text_features import hashing_vectorize


def _linearly_separable(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n).astype(bool)
    # feature 0 is a strong, feature 1 is pure noise.
    x0 = np.where(y, rng.normal(3.0, 0.5, n), rng.normal(-3.0, 0.5, n))
    x1 = rng.normal(0.0, 1.0, n)
    return np.column_stack([x0, x1]), y


def test_select_l2_by_cv_picks_from_grid_deterministically() -> None:
    x, y = _linearly_separable(120, seed=0)
    l2_a = select_l2_by_cv(x, y, seed=0)
    l2_b = select_l2_by_cv(x, y, seed=0)
    assert l2_a == l2_b  # deterministic given the same seed


def test_fit_domain_probe_then_evaluate_cell_separates_well() -> None:
    x_train, y_train = _linearly_separable(200, seed=1)
    x_test, y_test = _linearly_separable(80, seed=2)
    probe, l2 = fit_domain_probe(x_train, y_train, seed=0)
    assert l2 > 0
    result = evaluate_cell(probe, x_test, y_test)
    assert result.auroc > 0.9
    assert result.n == 80
    assert result.n_pos == int(y_test.sum())
    assert 0.0 <= result.ece <= 1.0


def test_evaluate_cell_flags_underpowered_below_gate() -> None:
    x_train, y_train = _linearly_separable(50, seed=1)
    x_small, y_small = _linearly_separable(20, seed=3)
    probe, _ = fit_domain_probe(x_train, y_train, seed=0)
    result = evaluate_cell(probe, x_small, y_small, power_gate=150)
    assert result.underpowered is True


def test_evaluate_cell_not_underpowered_when_positives_clear_gate() -> None:
    x_train, y_train = _linearly_separable(400, seed=1)
    x_big, y_big = _linearly_separable(400, seed=4)
    probe, _ = fit_domain_probe(x_train, y_train, seed=0)
    result = evaluate_cell(probe, x_big, y_big, power_gate=50)
    assert result.n_pos >= 50
    assert result.underpowered is False


def test_evaluate_cell_single_class_gives_nan_auroc_not_spurious_value() -> None:
    x_train, y_train = _linearly_separable(100, seed=1)
    probe, _ = fit_domain_probe(x_train, y_train, seed=0)
    x_single = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    y_single = np.array([False, False, False])
    result = evaluate_cell(probe, x_single, y_single)
    assert np.isnan(result.auroc)


def test_majority_class_baseline_reports_train_majority_and_eval_accuracy() -> None:
    y_train = np.array([False] * 80 + [True] * 20)  # majority = compliant
    y_eval = np.array([False] * 5 + [True] * 5)
    result = majority_class_baseline(y_train, y_eval)
    assert result.majority_label is False
    assert result.accuracy == 0.5  # gets all 5 negatives right, all 5 positives wrong
    assert result.n == 10
    assert result.n_pos == 5


def test_majority_class_baseline_empty_eval_is_nan_not_crash() -> None:
    y_train = np.array([True, False])
    y_eval = np.array([], dtype=bool)
    result = majority_class_baseline(y_train, y_eval)
    assert np.isnan(result.accuracy)


def test_hashing_features_feed_into_probe_end_to_end() -> None:
    texts_train = ["please cancel my order now"] * 30 + ["thanks for the help today"] * 30
    y_train = np.array([True] * 30 + [False] * 30)
    x_train = hashing_vectorize(texts_train, n_features=64)
    probe, _ = fit_domain_probe(x_train, y_train, seed=0)
    x_test = hashing_vectorize(
        ["please cancel my order now", "thanks for the help today"] * 5, n_features=64
    )
    y_test = np.array([True, False] * 5)
    result = evaluate_cell(probe, x_test, y_test)
    assert result.auroc > 0.8
