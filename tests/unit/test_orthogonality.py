from __future__ import annotations

import pytest

from bossyk_sandbox.scoring.orthogonality import (
    MembershipCell,
    StepMembership,
    orthogonality_table,
    wilson_interval,
)


def test_wilson_interval_bounds_a_50_percent_rate() -> None:
    low, high = wilson_interval(5, 10)
    assert low < 0.5 < high
    assert 0.0 <= low
    assert high <= 1.0


def test_wilson_interval_empty_sample_is_zero() -> None:
    assert wilson_interval(0, 0) == (0.0, 0.0)


def test_wilson_interval_narrows_with_more_data() -> None:
    low_small, high_small = wilson_interval(5, 10)
    low_large, high_large = wilson_interval(500, 1000)
    assert (high_large - low_large) < (high_small - low_small)


def test_orthogonality_table_has_all_eight_cells() -> None:
    records = [
        StepMembership(step_id="s1", drift_fires=True, policy_fires=False, outcome_violation=True)
    ]

    table = orthogonality_table(records)

    assert len(table) == 8
    cells = {row.cell for row in table}
    assert MembershipCell(True, False, True) in cells
    assert MembershipCell(False, False, False) in cells


def test_orthogonality_table_counts_known_membership() -> None:
    records = [
        StepMembership(step_id="s1", drift_fires=True, policy_fires=True, outcome_violation=True),
        StepMembership(step_id="s2", drift_fires=True, policy_fires=True, outcome_violation=True),
        StepMembership(
            step_id="s3", drift_fires=False, policy_fires=False, outcome_violation=False
        ),
        StepMembership(step_id="s4", drift_fires=True, policy_fires=False, outcome_violation=False),
    ]

    table = orthogonality_table(records)
    by_cell = {row.cell: row for row in table}

    all_true = by_cell[MembershipCell(True, True, True)]
    assert all_true.count == 2
    assert all_true.proportion == pytest.approx(0.5)

    all_false = by_cell[MembershipCell(False, False, False)]
    assert all_false.count == 1

    drift_only = by_cell[MembershipCell(True, False, False)]
    assert drift_only.count == 1

    unseen = by_cell[MembershipCell(False, True, False)]
    assert unseen.count == 0
    assert unseen.ci_low == 0.0


def test_orthogonality_table_empty_input_all_zero() -> None:
    table = orthogonality_table([])
    assert all(row.count == 0 for row in table)
    assert all(row.proportion == 0.0 for row in table)
