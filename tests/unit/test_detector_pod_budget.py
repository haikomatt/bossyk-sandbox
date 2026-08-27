"""Unit tests for bossyk_sandbox.detector.pod_budget -- the hermetic budget
arithmetic backing the pod runner's auto-terminate safety rail."""

from __future__ import annotations

import pytest

from bossyk_sandbox.detector.pod_budget import (
    HARD_CAP_USD,
    MAX_WALL_SECONDS,
    cost_estimate_usd,
    remaining_seconds,
    should_terminate,
    wall_budget_seconds,
)


def test_wall_budget_is_capped_by_the_six_hour_floor_when_dollar_cap_is_generous() -> None:
    budget = wall_budget_seconds(hard_cap_usd=1000.0, hourly_usd=0.44)
    assert budget == MAX_WALL_SECONDS


def test_wall_budget_is_tighter_when_dollar_cap_binds() -> None:
    budget = wall_budget_seconds(hard_cap_usd=1.0, hourly_usd=0.44)
    assert budget < MAX_WALL_SECONDS
    assert budget == int(1.0 / 0.44 * 3600)


def test_default_wall_budget_is_under_three_hours_worth_of_the_dollar_cap() -> None:
    # $15 / $0.44/hr ~= 34h, so the 6h floor is what actually binds by default.
    budget = wall_budget_seconds()
    assert budget == MAX_WALL_SECONDS
    assert cost_estimate_usd(budget) < 3.0  # "under $3 at A40 rates"


def test_cost_estimate_scales_linearly_with_elapsed_time() -> None:
    assert cost_estimate_usd(3600.0, hourly_usd=0.44) == pytest.approx(0.44)
    assert cost_estimate_usd(7200.0, hourly_usd=0.44) == pytest.approx(0.88)


def test_cost_estimate_rejects_negative_elapsed() -> None:
    with pytest.raises(ValueError):
        cost_estimate_usd(-1.0)


def test_wall_budget_rejects_non_positive_inputs() -> None:
    with pytest.raises(ValueError):
        wall_budget_seconds(hard_cap_usd=0.0)
    with pytest.raises(ValueError):
        wall_budget_seconds(hourly_usd=0.0)


def test_should_terminate_false_well_within_budget() -> None:
    decision = should_terminate(60.0)
    assert decision.terminate is False
    assert "within budget" in decision.reason


def test_should_terminate_true_when_wall_clock_exceeded() -> None:
    decision = should_terminate(MAX_WALL_SECONDS + 1)
    assert decision.terminate is True
    assert "wall-clock budget reached" in decision.reason


def test_should_terminate_true_when_dollar_cap_exceeded_even_under_wall_clock() -> None:
    # A tiny wall budget forces the dollar check to be the one that fires
    # first in isolation -- exercise it directly with a low max_wall_seconds
    # so wall-clock does NOT also trip at the same instant.
    decision = should_terminate(3600.0, hard_cap_usd=0.10, hourly_usd=0.44, max_wall_seconds=10**9)
    assert decision.terminate is True
    assert "dollar cap reached" in decision.reason


def test_should_terminate_boundary_is_inclusive() -> None:
    budget = wall_budget_seconds()
    decision = should_terminate(float(budget))
    assert decision.terminate is True


def test_remaining_seconds_is_never_negative_past_budget() -> None:
    assert remaining_seconds(MAX_WALL_SECONDS + 1000) == 0.0


def test_remaining_seconds_counts_down() -> None:
    budget = wall_budget_seconds()
    assert remaining_seconds(0.0) == float(budget)
    assert remaining_seconds(budget / 2) == pytest.approx(budget / 2)


def test_defaults_match_documented_constants() -> None:
    assert HARD_CAP_USD == 15.0
    assert MAX_WALL_SECONDS == 6 * 3600
