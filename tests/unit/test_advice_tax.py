from __future__ import annotations

from bossyk_sandbox.advice.tax import (
    ADDITIONAL_RATE_LOWER_GBP,
    BASIC_RATE_UPPER_GBP,
    TaxBand,
    contribution_headroom_band,
    tax_band,
)


def test_tax_band_at_basic_upper_boundary_is_basic() -> None:
    assert tax_band(BASIC_RATE_UPPER_GBP) is TaxBand.BASIC


def test_tax_band_just_above_basic_upper_boundary_is_higher() -> None:
    assert tax_band(BASIC_RATE_UPPER_GBP + 1) is TaxBand.HIGHER


def test_tax_band_at_additional_lower_boundary_is_higher() -> None:
    assert tax_band(ADDITIONAL_RATE_LOWER_GBP) is TaxBand.HIGHER


def test_tax_band_just_above_additional_lower_boundary_is_additional() -> None:
    assert tax_band(ADDITIONAL_RATE_LOWER_GBP + 1) is TaxBand.ADDITIONAL


def test_tax_band_zero_income_is_basic() -> None:
    assert tax_band(0) is TaxBand.BASIC


def test_contribution_headroom_band_is_constant_within_a_band() -> None:
    # Two incomes in the same band must yield the same headroom band -- the
    # tool is derived from the band only, never from the raw income, so it
    # cannot be used to narrow an income estimate any further than the band
    # predicate already allows.
    assert contribution_headroom_band(20_000) == contribution_headroom_band(50_270)


def test_contribution_headroom_band_differs_across_bands() -> None:
    basic = contribution_headroom_band(20_000)
    higher = contribution_headroom_band(80_000)
    additional = contribution_headroom_band(200_000)

    assert len({basic, higher, additional}) == 3
