"""Hermetic tests for interp.correlate -- pure statistics, no I/O, no model."""

from __future__ import annotations

import math

import pytest

from bossyk_sandbox.interp.correlate import (
    auroc,
    correlate_uncertainty,
    format_separation_table,
)
from bossyk_sandbox.interp.logprob_metrics import StepUncertainty


def _step(
    *,
    n_tokens: int = 3,
    max_surprisal: float = 0.0,
    mean_surprisal: float = 0.0,
    mean_entropy: float = 0.0,
    min_margin: float = 0.0,
) -> StepUncertainty:
    return StepUncertainty(
        n_tokens=n_tokens,
        mean_surprisal=mean_surprisal,
        max_surprisal=max_surprisal,
        mean_entropy=mean_entropy,
        min_margin=min_margin,
    )


def test_auroc_perfect_separation_positive_higher_is_one() -> None:
    assert auroc([1.0, 2.0, 3.0, 4.0], [False, False, True, True]) == 1.0


def test_auroc_perfect_separation_positive_lower_is_zero() -> None:
    assert auroc([1.0, 2.0, 3.0, 4.0], [True, True, False, False]) == 0.0


def test_auroc_all_ties_is_half() -> None:
    assert auroc([1.0, 1.0, 2.0, 2.0], [False, True, False, True]) == 0.5


def test_auroc_empty_class_is_nan() -> None:
    assert math.isnan(auroc([1.0, 2.0], [True, True]))
    assert math.isnan(auroc([1.0, 2.0], [False, False]))


def test_auroc_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="same length"):
        auroc([1.0, 2.0], [True])


def test_correlate_detects_higher_surprisal_on_violations() -> None:
    pairs = [
        (_step(max_surprisal=0.1), False),
        (_step(max_surprisal=0.2), False),
        (_step(max_surprisal=3.0), True),
        (_step(max_surprisal=4.0), True),
    ]
    rows = {r.feature: r for r in correlate_uncertainty(pairs)}
    assert rows["max_surprisal"].auroc == 1.0  # violations cleanly higher
    assert rows["max_surprisal"].n_violation == 2
    assert rows["max_surprisal"].n_compliant == 2
    assert rows["max_surprisal"].mean_violation == pytest.approx(3.5)
    assert rows["max_surprisal"].mean_compliant == pytest.approx(0.15)


def test_correlate_captures_reversed_direction_for_margin() -> None:
    # Violations are near-ties: SMALLER min_margin -> AUROC below 0.5.
    pairs = [
        (_step(min_margin=2.0), False),
        (_step(min_margin=1.8), False),
        (_step(min_margin=0.1), True),
        (_step(min_margin=0.2), True),
    ]
    rows = {r.feature: r for r in correlate_uncertainty(pairs)}
    assert rows["min_margin"].auroc == 0.0


def test_correlate_excludes_zero_token_steps() -> None:
    # A pure tool-call turn (n_tokens == 0) must not enter the statistics.
    pairs = [
        (_step(n_tokens=0, max_surprisal=99.0), True),  # excluded despite the label
        (_step(max_surprisal=0.1), False),
        (_step(max_surprisal=3.0), True),
    ]
    rows = {r.feature: r for r in correlate_uncertainty(pairs)}
    assert rows["max_surprisal"].n_violation == 1
    assert rows["max_surprisal"].n_compliant == 1


def test_correlate_all_one_class_yields_nan_auroc() -> None:
    pairs = [(_step(max_surprisal=1.0), True), (_step(max_surprisal=2.0), True)]
    rows = {r.feature: r for r in correlate_uncertainty(pairs)}
    assert math.isnan(rows["max_surprisal"].auroc)


def test_format_separation_table_renders_markdown() -> None:
    pairs = [(_step(max_surprisal=0.1), False), (_step(max_surprisal=3.0), True)]
    table = format_separation_table(correlate_uncertainty(pairs))
    assert table.startswith("| feature |")
    assert "max_surprisal" in table
    assert "AUROC" in table
