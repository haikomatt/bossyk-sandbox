"""Unit tests for bossyk_sandbox.detector.temperature."""

from __future__ import annotations

import numpy as np
import pytest

from bossyk_sandbox.detector.temperature import (
    apply_temperature,
    fit_temperature,
    negative_log_likelihood,
)


def test_apply_temperature_t1_is_plain_sigmoid() -> None:
    logits = np.array([-2.0, 0.0, 2.0])
    p = apply_temperature(logits, 1.0)
    expected = 1.0 / (1.0 + np.exp(-logits))
    assert np.allclose(p, expected)


def test_apply_temperature_high_t_pulls_toward_half() -> None:
    logits = np.array([5.0, -5.0])
    p = apply_temperature(logits, 100.0)
    assert np.allclose(p, 0.5, atol=0.05)


def test_apply_temperature_rejects_nonpositive() -> None:
    with pytest.raises(ValueError):
        apply_temperature(np.array([1.0]), 0.0)


def test_fit_temperature_on_overconfident_logits_scales_up() -> None:
    # Correct-direction, wildly overconfident logits, but with a slice of
    # label noise so the model IS sometimes confidently wrong (a perfectly
    # separable, always-correct slice has no calibration to fix -- sharper
    # is strictly better there, T->t_min; this was verified directly and
    # the test below covers it as the "well separated" case instead).
    # With noise, an overconfident wrong prediction costs a lot of NLL, so
    # fitting T should push it above 1 to soften the probabilities.
    rng = np.random.default_rng(0)
    n = 400
    y_true = rng.integers(0, 2, size=n).astype(bool)
    noisy = rng.uniform(size=n) < 0.15
    y_observed = np.where(noisy, ~y_true, y_true)  # logits reflect y_true; labels are noisy
    logits = np.where(y_true, rng.normal(20.0, 1.0, n), rng.normal(-20.0, 1.0, n))
    t = fit_temperature(logits, y_observed)
    assert t > 1.0
    nll_fit = negative_log_likelihood(logits, y_observed, t)
    nll_raw = negative_log_likelihood(logits, y_observed, 1.0)
    assert nll_fit <= nll_raw


def test_fit_temperature_on_perfectly_separable_logits_sharpens_not_softens() -> None:
    # No noise, huge-margin correct logits: NLL at T=1 is already ~0, so the
    # NLL-minimising fit sharpens further (T < 1) rather than softening --
    # there is nothing mis-calibrated to fix, sharper is strictly better.
    rng = np.random.default_rng(0)
    n = 400
    y = rng.integers(0, 2, size=n).astype(bool)
    logits = np.where(y, rng.normal(20.0, 1.0, n), rng.normal(-20.0, 1.0, n))
    t = fit_temperature(logits, y)
    assert t < 1.0


def test_fit_temperature_well_calibrated_logits_stays_near_one() -> None:
    rng = np.random.default_rng(1)
    n = 2000
    true_p = rng.uniform(0.05, 0.95, n)
    y = rng.uniform(size=n) < true_p
    logits = np.log(true_p / (1 - true_p))  # exactly the logit of the true prob
    t = fit_temperature(logits, y)
    assert 0.7 <= t <= 1.4


def test_fit_temperature_deterministic() -> None:
    rng = np.random.default_rng(2)
    logits = rng.normal(size=100)
    y = rng.integers(0, 2, size=100).astype(bool)
    t1 = fit_temperature(logits, y)
    t2 = fit_temperature(logits, y)
    assert t1 == t2


def test_fit_temperature_degenerate_tiny_slice_returns_one() -> None:
    assert fit_temperature(np.array([]), np.array([], dtype=bool)) == 1.0
    assert fit_temperature(np.array([1.0]), np.array([True])) == 1.0


def test_negative_log_likelihood_perfect_confident_correct_is_near_zero() -> None:
    logits = np.array([50.0, -50.0, 50.0, -50.0])
    y = np.array([True, False, True, False])
    nll = negative_log_likelihood(logits, y, 1.0)
    assert nll < 1e-6


def test_negative_log_likelihood_confident_wrong_is_large_but_finite() -> None:
    logits = np.array([50.0])
    y = np.array([False])
    nll = negative_log_likelihood(logits, y, 1.0)
    assert nll > 10.0
    assert np.isfinite(nll)


def test_negative_log_likelihood_empty_is_nan() -> None:
    assert np.isnan(negative_log_likelihood(np.array([]), np.array([], dtype=bool), 1.0))
