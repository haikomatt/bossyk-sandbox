"""Unit tests for the Wilson score interval (Amendment 2b(i) base-rate CI)."""

from __future__ import annotations

import math

from bossyk_sandbox.detector.base_rate import wilson_interval


def test_wilson_interval_matches_known_textbook_value() -> None:
    # n=100, n_pos=50 -> classic textbook Wilson 95% CI approx (0.404, 0.596).
    ci = wilson_interval(50, 100)
    assert abs(ci.rate - 0.5) < 1e-12
    assert math.isclose(ci.lower, 0.4038, abs_tol=1e-3)
    assert math.isclose(ci.upper, 0.5962, abs_tol=1e-3)


def test_wilson_interval_bounds_within_unit_interval() -> None:
    ci = wilson_interval(1, 1000)
    assert 0.0 <= ci.lower <= ci.rate <= ci.upper <= 1.0


def test_wilson_interval_zero_n_is_nan_not_crash() -> None:
    ci = wilson_interval(0, 0)
    assert math.isnan(ci.rate)
    assert math.isnan(ci.lower)
    assert math.isnan(ci.upper)


def test_wilson_interval_all_positive_upper_bounded_at_one() -> None:
    ci = wilson_interval(50, 50)
    assert ci.rate == 1.0
    assert ci.upper <= 1.0
    assert ci.lower > 0.0


def test_wilson_interval_all_negative_lower_bounded_at_zero() -> None:
    ci = wilson_interval(0, 50)
    assert ci.rate == 0.0
    assert ci.lower >= 0.0
    assert ci.upper < 1.0


def test_wilson_interval_narrows_with_more_data_same_rate() -> None:
    small = wilson_interval(20, 100)
    large = wilson_interval(200, 1000)
    assert (large.upper - large.lower) < (small.upper - small.lower)


def test_wilson_interval_rejects_invalid_counts() -> None:
    try:
        wilson_interval(10, 5)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for n_pos > n")
