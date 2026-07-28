from __future__ import annotations

import pytest

from bossyk_sandbox.scoring.latency import LatencySummary
from bossyk_sandbox.scoring.latency_budget import (
    ACTION_EXEC_LABEL,
    BudgetComparison,
    budget_comparisons,
    latency_budget,
)

# A slow policy judge: round numbers so every ratio is hand-checkable. mean 26s,
# p50 42s, p95 42s (nearest-rank collapses p50/p95 on this tiny distribution --
# matches the real committed weak-structural run's shape, n small).
SLOW_POLICY = LatencySummary(
    instrument="policy",
    count=4,
    mean_s=26.0,
    p50_s=42.0,
    p95_s=42.0,
    max_s=42.0,
)

# A hypothetical already-fast detector: mean 0.1s, well under a 0.5s budget.
FAST_DETECTOR = LatencySummary(
    instrument="fast",
    count=4,
    mean_s=0.1,
    p50_s=0.1,
    p95_s=0.2,
    max_s=0.2,
)

UX_BUDGETS = {"ux_250ms": 0.25, "ux_500ms": 0.5, "ux_1s": 1.0}
ACTION_EXEC_S = 0.5


def _by_label(comparisons: list[BudgetComparison]) -> dict[str, BudgetComparison]:
    return {c.budget_label: c for c in comparisons}


def test_budget_comparisons_reports_action_exec_floor_first_then_ux_budgets() -> None:
    comparisons = budget_comparisons(
        SLOW_POLICY, action_exec_s=ACTION_EXEC_S, ux_budgets_s=UX_BUDGETS
    )

    labels = [c.budget_label for c in comparisons]
    assert labels == [ACTION_EXEC_LABEL, "ux_250ms", "ux_500ms", "ux_1s"]
    assert all(c.instrument == "policy" for c in comparisons)


def test_budget_comparisons_computes_speedup_needed_as_latency_over_budget() -> None:
    by_label = _by_label(
        budget_comparisons(SLOW_POLICY, action_exec_s=ACTION_EXEC_S, ux_budgets_s=UX_BUDGETS)
    )

    # 500ms UX budget: mean 26 / 0.5 = 52x, p95 42 / 0.5 = 84x.
    ux500 = by_label["ux_500ms"]
    assert ux500.budget_s == 0.5
    assert ux500.speedup_needed_mean == pytest.approx(52.0)
    assert ux500.speedup_needed_p95 == pytest.approx(84.0)

    # 250ms is twice as demanding as 500ms.
    ux250 = by_label["ux_250ms"]
    assert ux250.speedup_needed_mean == pytest.approx(104.0)
    assert ux250.speedup_needed_p95 == pytest.approx(168.0)

    # 1s budget: mean 26 / 1.0 = 26x.
    assert by_label["ux_1s"].speedup_needed_mean == pytest.approx(26.0)


def test_budget_comparisons_action_exec_floor_uses_the_measured_exec_time() -> None:
    by_label = _by_label(
        budget_comparisons(SLOW_POLICY, action_exec_s=ACTION_EXEC_S, ux_budgets_s=UX_BUDGETS)
    )

    floor = by_label[ACTION_EXEC_LABEL]
    assert floor.budget_s == ACTION_EXEC_S
    # 26 / 0.5 = 52x to be free within the action's own execution cost.
    assert floor.speedup_needed_mean == pytest.approx(52.0)


def test_budget_comparisons_computes_lateness_percentiles_shifted_by_budget() -> None:
    by_label = _by_label(
        budget_comparisons(SLOW_POLICY, action_exec_s=ACTION_EXEC_S, ux_budgets_s=UX_BUDGETS)
    )

    ux500 = by_label["ux_500ms"]
    # lateness = max(0, latency_percentile - budget): p50 42 - 0.5, p95 42 - 0.5.
    assert ux500.lateness_p50_s == pytest.approx(41.5)
    assert ux500.lateness_p95_s == pytest.approx(41.5)


def test_budget_comparisons_flags_prevented_in_time_when_mean_fits_the_budget() -> None:
    by_label = _by_label(
        budget_comparisons(FAST_DETECTOR, action_exec_s=ACTION_EXEC_S, ux_budgets_s=UX_BUDGETS)
    )

    # mean 0.1s fits a 0.5s budget -> prevented in time, no lateness, speedup < 1.
    ux500 = by_label["ux_500ms"]
    assert ux500.prevented_in_time_at_mean is True
    assert ux500.lateness_p50_s == pytest.approx(0.0)
    assert ux500.lateness_p95_s == pytest.approx(0.0)
    assert ux500.speedup_needed_mean == pytest.approx(0.2)

    # ...but mean 0.1s does NOT fit a 0.25s budget? 0.1 <= 0.25 -> still fits.
    assert by_label["ux_250ms"].prevented_in_time_at_mean is True


def test_budget_comparisons_flags_too_late_when_mean_exceeds_the_budget() -> None:
    by_label = _by_label(
        budget_comparisons(SLOW_POLICY, action_exec_s=ACTION_EXEC_S, ux_budgets_s=UX_BUDGETS)
    )

    assert by_label["ux_500ms"].prevented_in_time_at_mean is False
    assert by_label["ux_1s"].prevented_in_time_at_mean is False


def test_budget_comparisons_omits_the_action_exec_floor_when_none() -> None:
    # A run whose every harmful action the gate blocked pre-execution has no
    # completed tool call to time -> no action-exec floor -> only the UX
    # speedups, which need detection latency alone.
    comparisons = budget_comparisons(SLOW_POLICY, action_exec_s=None, ux_budgets_s=UX_BUDGETS)

    assert [c.budget_label for c in comparisons] == ["ux_250ms", "ux_500ms", "ux_1s"]
    assert _by_label(comparisons)["ux_500ms"].speedup_needed_mean == pytest.approx(52.0)


def test_budget_comparisons_rejects_non_positive_budgets() -> None:
    with pytest.raises(ValueError):
        budget_comparisons(SLOW_POLICY, action_exec_s=0.0, ux_budgets_s=UX_BUDGETS)

    with pytest.raises(ValueError):
        budget_comparisons(SLOW_POLICY, action_exec_s=ACTION_EXEC_S, ux_budgets_s={"bad": -1.0})


def test_latency_budget_maps_over_every_instrument() -> None:
    summaries = {"policy": SLOW_POLICY, "fast": FAST_DETECTOR}

    result = latency_budget(summaries, action_exec_s=ACTION_EXEC_S, ux_budgets_s=UX_BUDGETS)

    assert set(result.keys()) == {"policy", "fast"}
    assert [c.budget_label for c in result["policy"]] == [
        ACTION_EXEC_LABEL,
        "ux_250ms",
        "ux_500ms",
        "ux_1s",
    ]
    # The mapping preserves per-instrument latency: policy p95 84x @ 500ms.
    policy_500 = _by_label(result["policy"])["ux_500ms"]
    assert policy_500.speedup_needed_p95 == pytest.approx(84.0)
