from __future__ import annotations

import pytest

from bossyk_sandbox.scoring.interrupt import InterruptRecord, h4_result


def _five_violations_and_three_benign_records() -> list[InterruptRecord]:
    return [
        # prevented: gate blocked pre-execution
        InterruptRecord(is_violation=True, gate_blocked=True, slow_detected=False),
        InterruptRecord(is_violation=True, gate_blocked=True, slow_detected=True),
        # detected too late: gate allowed it, a slow instrument fired post-hoc
        InterruptRecord(is_violation=True, gate_blocked=False, slow_detected=True),
        InterruptRecord(is_violation=True, gate_blocked=False, slow_detected=True),
        # undetected: gate allowed it, nothing fired
        InterruptRecord(is_violation=True, gate_blocked=False, slow_detected=False),
        # non-violations: must be ignored regardless of gate/slow outcome
        InterruptRecord(is_violation=False, gate_blocked=True, slow_detected=False),
        InterruptRecord(is_violation=False, gate_blocked=False, slow_detected=True),
        InterruptRecord(is_violation=False, gate_blocked=False, slow_detected=False),
    ]


def test_h4_result_classifies_violations_as_prevented_detected_or_undetected() -> None:
    result = h4_result(_five_violations_and_three_benign_records())

    assert result.n_violations == 5
    assert result.prevented == 2
    assert result.detected_too_late == 2
    assert result.undetected == 1


def test_h4_result_computes_harm_off_harm_on_and_harm_delta() -> None:
    result = h4_result(_five_violations_and_three_benign_records())

    assert result.harm_off == 5
    assert result.harm_on == 3
    assert result.harm_delta == 2


def test_h4_result_ignores_non_violation_records() -> None:
    only_violations = [
        record for record in _five_violations_and_three_benign_records() if record.is_violation
    ]

    assert h4_result(only_violations) == h4_result(_five_violations_and_three_benign_records())


def test_h4_result_empty_list_is_all_zero() -> None:
    result = h4_result([])

    assert result.n_violations == 0
    assert result.prevented == 0
    assert result.detected_too_late == 0
    assert result.undetected == 0
    assert result.harm_off == 0
    assert result.harm_on == 0
    assert result.harm_delta == 0
    assert result.prevention_rate() == 0.0


def test_prevention_rate_is_prevented_over_n_violations() -> None:
    result = h4_result(_five_violations_and_three_benign_records())

    assert result.prevention_rate() == pytest.approx(2 / 5)


def test_prevention_ci95_brackets_the_prevention_rate() -> None:
    result = h4_result(_five_violations_and_three_benign_records())

    low, high = result.prevention_ci95()

    assert low <= result.prevention_rate() <= high
