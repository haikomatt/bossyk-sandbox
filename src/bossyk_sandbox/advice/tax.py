"""UK income-tax band thresholds and the derived (non-raw) values built on
top of them for the advice domain (bossyk-sandbox privacy/minimisation
demonstrator, Phase 1). Figures are the public rest-of-UK (England/Wales/
Northern Ireland) 2024/25 thresholds, used here purely as fixed demo
constants -- this module gives no tax advice.
"""

from __future__ import annotations

from enum import StrEnum

BASIC_RATE_UPPER_GBP = 50_270
ADDITIONAL_RATE_LOWER_GBP = 125_140


class TaxBand(StrEnum):
    BASIC = "basic"
    HIGHER = "higher"
    ADDITIONAL = "additional"


def tax_band(annual_income_gbp: float) -> TaxBand:
    """Bands `annual_income_gbp` into basic/higher/additional. A boundary
    value sits in the LOWER band -- income exactly at a threshold has not
    yet crossed it: `BASIC_RATE_UPPER_GBP` itself is still `BASIC`,
    `ADDITIONAL_RATE_LOWER_GBP` itself is still `HIGHER`."""
    if annual_income_gbp <= BASIC_RATE_UPPER_GBP:
        return TaxBand.BASIC
    if annual_income_gbp <= ADDITIONAL_RATE_LOWER_GBP:
        return TaxBand.HIGHER
    return TaxBand.ADDITIONAL


# Deliberately coarse: three bands, one per tax band, nothing finer. Kept as
# an explicit table (not a formula over income) so it's obvious by
# inspection that no raw income value ever feeds it.
_HEADROOM_BAND_BY_TAX_BAND: dict[TaxBand, str] = {
    TaxBand.BASIC: "high",
    TaxBand.HIGHER: "medium",
    TaxBand.ADDITIONAL: "low",
}


def contribution_headroom_band(annual_income_gbp: float) -> str:
    """A deliberately coarse pension-contribution-headroom indicator,
    derived from the tax band ALONE, never from the raw income -- so this
    tool can never narrow an income estimate beyond what the band predicate
    (`tax_band`) already reveals."""
    return _HEADROOM_BAND_BY_TAX_BAND[tax_band(annual_income_gbp)]
