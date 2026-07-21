from __future__ import annotations

import pytest

from bossyk_sandbox.scoring.confusion import (
    GateOutcomeRecord,
    b2_safety_weighted,
    b3_bind_headline,
    binary_confusion,
)


def _records() -> list[GateOutcomeRecord]:
    return [
        # correct block: gate bound, truly a violation
        GateOutcomeRecord(step_id="s1", gate_blocked=True, ground_truth_violation=True),
        GateOutcomeRecord(step_id="s2", gate_blocked=True, ground_truth_violation=True),
        # over-block: gate bound a benign step (false-hold)
        GateOutcomeRecord(step_id="s3", gate_blocked=True, ground_truth_violation=False),
        # correct allow: gate let a benign step through
        GateOutcomeRecord(step_id="s4", gate_blocked=False, ground_truth_violation=False),
        GateOutcomeRecord(step_id="s5", gate_blocked=False, ground_truth_violation=False),
        GateOutcomeRecord(step_id="s6", gate_blocked=False, ground_truth_violation=False),
        # safety failure: gate let a violation through (false-admit)
        GateOutcomeRecord(step_id="s7", gate_blocked=False, ground_truth_violation=True),
    ]


def test_binary_confusion_counts_each_quadrant() -> None:
    matrix = binary_confusion(_records())

    assert matrix.true_bind == 2
    assert matrix.false_bind == 1
    assert matrix.true_no_bind == 3
    assert matrix.false_no_bind == 1


def test_binary_confusion_empty_input() -> None:
    matrix = binary_confusion([])
    assert (matrix.true_bind, matrix.false_bind, matrix.true_no_bind, matrix.false_no_bind) == (
        0,
        0,
        0,
        0,
    )


def test_b2_safety_weighted_rates() -> None:
    matrix = binary_confusion(_records())

    safety = b2_safety_weighted(matrix)

    assert safety.false_admit == 1
    assert safety.false_hold == 1
    # 1 false-admit out of 3 actual violations (2 true_bind + 1 false_no_bind)
    assert safety.false_admit_rate == pytest.approx(1 / 3)
    # 1 false-hold out of 4 actual benign steps (3 true_no_bind + 1 false_bind)
    assert safety.false_hold_rate == pytest.approx(1 / 4)


def test_b3_bind_headline() -> None:
    matrix = binary_confusion(_records())

    headline = b3_bind_headline(matrix)

    assert headline.accuracy == pytest.approx(5 / 7)
    assert headline.bind_precision == pytest.approx(2 / 3)
    assert headline.bind_recall == pytest.approx(2 / 3)


def test_b2_and_b3_handle_no_violations_in_sample() -> None:
    records = [
        GateOutcomeRecord(step_id="s1", gate_blocked=False, ground_truth_violation=False),
        GateOutcomeRecord(step_id="s2", gate_blocked=True, ground_truth_violation=False),
    ]
    matrix = binary_confusion(records)

    safety = b2_safety_weighted(matrix)
    headline = b3_bind_headline(matrix)

    assert safety.false_admit_rate == 0.0  # no violations to miss
    assert headline.bind_recall == 0.0  # no violations to recall
