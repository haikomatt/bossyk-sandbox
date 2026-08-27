"""Hermetic tests for `bossyk_sandbox.interp.size_calc` (hermetic diversity
fix, item 4: the run-2 sizing calculator). Pure arithmetic -- no network, no
LLM, no filesystem.

Feeds the run-2 authorization ask (phase-detector-training-step2-datagen.md
Issues & Fixes / Part B run 1 fix plan, step 3): given a per-domain MEASURED
unique-rate and violation base rate (from a small calibration sample) plus
the real per-call cost, compute the call volume and $ needed to reach the
dossier's ~150-300-positives/OOD-cell power floor
(a-fine-tuned-violation-detector-transfers-cross-domain.md).
"""

from __future__ import annotations

import math

import pytest

from bossyk_sandbox.interp.size_calc import SizeEstimate, size_run


def test_size_run_basic_arithmetic() -> None:
    # 150 target positives, violation_rate 0.10 (over UNIQUE decisions) ->
    # need 1500 unique decisions; unique_rate 0.5 -> need 3000 calls;
    # cost_per_call_usd 0.002 -> $6.00.
    estimate = size_run(
        domain="retail",
        unique_rate=0.5,
        violation_rate=0.10,
        cost_per_call_usd=0.002,
        target_positives=150,
    )
    assert isinstance(estimate, SizeEstimate)
    assert estimate.domain == "retail"
    assert estimate.target_positives == 150
    assert estimate.unique_decisions_needed == 1500
    assert estimate.calls_needed == 3000
    assert estimate.estimated_cost_usd == pytest.approx(6.0)


def test_size_run_rounds_up_fractional_calls_and_decisions() -> None:
    # violation_rate 0.03 -> 150/0.03 = 5000 exactly; pick a rate that
    # doesn't divide evenly to prove ceiling (not floor/round) behaviour.
    estimate = size_run(
        domain="retail",
        unique_rate=0.017,  # ~= run-1's measured retail unique rate
        violation_rate=0.04,  # ~= run-1's measured retail unique-level violation rate
        cost_per_call_usd=0.002,
        target_positives=150,
    )
    unique_needed = math.ceil(150 / 0.04)
    calls_needed = math.ceil(unique_needed / 0.017)
    assert estimate.unique_decisions_needed == unique_needed
    assert estimate.calls_needed == calls_needed
    assert estimate.estimated_cost_usd == pytest.approx(calls_needed * 0.002)


def test_size_run_higher_target_needs_more_calls() -> None:
    low = size_run(
        domain="retail",
        unique_rate=0.1,
        violation_rate=0.1,
        cost_per_call_usd=0.002,
        target_positives=150,
    )
    high = size_run(
        domain="retail",
        unique_rate=0.1,
        violation_rate=0.1,
        cost_per_call_usd=0.002,
        target_positives=300,
    )
    assert high.calls_needed > low.calls_needed
    assert high.estimated_cost_usd > low.estimated_cost_usd


@pytest.mark.parametrize(
    "kwargs",
    [
        {"unique_rate": 0.0},
        {"unique_rate": 1.5},
        {"unique_rate": -0.1},
        {"violation_rate": 0.0},
        {"violation_rate": 1.5},
        {"cost_per_call_usd": -0.001},
        {"target_positives": 0},
        {"target_positives": -5},
    ],
)
def test_size_run_rejects_out_of_range_inputs(kwargs: dict[str, float]) -> None:
    base = {
        "domain": "retail",
        "unique_rate": 0.5,
        "violation_rate": 0.1,
        "cost_per_call_usd": 0.002,
        "target_positives": 150,
    }
    base.update(kwargs)
    with pytest.raises(ValueError):
        size_run(**base)  # type: ignore[arg-type]
