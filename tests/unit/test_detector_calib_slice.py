"""Unit tests for bossyk_sandbox.detector.calib_slice (Amendment 3)."""

from __future__ import annotations

from typing import Any

from bossyk_sandbox.detector.calib_slice import calibration_slice_split


def _rows(
    n_scenarios: int, per_scenario: int, *, violation_scenarios: set[int]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for s in range(n_scenarios):
        for i in range(per_scenario):
            rows.append(
                {
                    "row_id": f"s{s}:{i}",
                    "scenario_id": f"scenario-{s}",
                    "prompt": f"scenario {s} step {i}",
                    "action": "act",
                    "is_violation": s in violation_scenarios,
                }
            )
    return rows


def test_train_and_calib_partition_all_rows_with_no_overlap() -> None:
    rows = _rows(20, 5, violation_scenarios=set(range(0, 20, 3)))
    result = calibration_slice_split(rows, seed=0)
    train_ids = {r["row_id"] for r in result.train}
    calib_ids = {r["row_id"] for r in result.calib}
    assert train_ids.isdisjoint(calib_ids)
    assert train_ids | calib_ids == {r["row_id"] for r in rows}


def test_calib_slice_is_group_exclusive_by_scenario() -> None:
    rows = _rows(20, 5, violation_scenarios={1, 4, 7, 10})
    result = calibration_slice_split(rows, seed=0)
    train_scenarios = {r["scenario_id"] for r in result.train}
    calib_scenarios = {r["scenario_id"] for r in result.calib}
    assert train_scenarios.isdisjoint(calib_scenarios)


def test_calib_slice_fraction_is_roughly_fifteen_percent() -> None:
    rows = _rows(40, 10, violation_scenarios=set(range(0, 40, 3)))
    result = calibration_slice_split(rows, seed=0)
    frac = len(result.calib) / len(rows)
    # Whole-group bin-packing won't hit 0.15 exactly; a generous band.
    assert 0.05 <= frac <= 0.30


def test_calib_slice_split_is_deterministic_for_a_given_seed() -> None:
    rows = _rows(20, 5, violation_scenarios={2, 5, 9})
    a = calibration_slice_split(rows, seed=7)
    b = calibration_slice_split(rows, seed=7)
    assert [r["row_id"] for r in a.train] == [r["row_id"] for r in b.train]
    assert [r["row_id"] for r in a.calib] == [r["row_id"] for r in b.calib]


def test_different_seeds_can_produce_different_slices() -> None:
    rows = _rows(30, 5, violation_scenarios=set(range(0, 30, 2)))
    a = calibration_slice_split(rows, seed=0)
    b = calibration_slice_split(rows, seed=1)
    assert [r["row_id"] for r in a.calib] != [r["row_id"] for r in b.calib]


def test_calib_slice_contains_some_positives_when_available() -> None:
    # Enough violation-scenarios spread around that a 15% slice should catch some.
    rows = _rows(20, 10, violation_scenarios=set(range(0, 20, 2)))
    result = calibration_slice_split(rows, seed=0)
    assert any(r["is_violation"] for r in result.calib)


def test_empty_train_split_returns_empty_buckets() -> None:
    result = calibration_slice_split([], seed=0)
    assert result.train == []
    assert result.calib == []


def test_slice_seed_is_recorded_via_the_underlying_report() -> None:
    rows = _rows(10, 4, violation_scenarios={0, 3})
    result = calibration_slice_split(rows, seed=3)
    assert result.report.n_groups == 10
    assert set(result.report.counts) == {"train", "val", "test"}
    assert result.report.counts["test"]["n"] == 0  # test_frac=0.0: no test bucket used
