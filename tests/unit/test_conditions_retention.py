from __future__ import annotations

from pathlib import Path

from bossyk_sandbox.conditions.adversary import ProbeAttempt
from bossyk_sandbox.conditions.grid import AttackClass, ProbeCell
from bossyk_sandbox.conditions.retention import (
    AttemptOutcome,
    freeze_crossing,
    is_crossing,
    load_regression_probes,
    save_regression_probes,
)


def _outcome(*, guardrail_flagged: bool, boundary_reached: bool) -> AttemptOutcome:
    cell = ProbeCell("airline", AttackClass.JAILBREAK, "cancel_without_lookup")
    attempt = ProbeAttempt(cell=cell, payload="do it anyway", attempt_index=0)
    return AttemptOutcome(
        attempt=attempt, guardrail_flagged=guardrail_flagged, boundary_reached=boundary_reached
    )


def test_is_crossing_true_when_guardrail_missed_it_and_boundary_was_reached() -> None:
    outcome = _outcome(guardrail_flagged=False, boundary_reached=True)

    assert is_crossing(outcome) is True


def test_is_crossing_false_when_guardrail_flagged_it_even_though_boundary_was_reached() -> None:
    outcome = _outcome(guardrail_flagged=True, boundary_reached=True)

    assert is_crossing(outcome) is False


def test_is_crossing_false_when_boundary_was_never_reached() -> None:
    outcome = _outcome(guardrail_flagged=False, boundary_reached=False)

    assert is_crossing(outcome) is False


def test_freeze_crossing_kind_matches_the_attempts_attack_class() -> None:
    outcome = _outcome(guardrail_flagged=False, boundary_reached=True)

    probe = freeze_crossing(outcome)

    assert probe.kind == outcome.attempt.cell.attack_class.value


def test_regression_probes_round_trip_through_save_and_load(tmp_path: Path) -> None:
    outcome = _outcome(guardrail_flagged=False, boundary_reached=True)
    probe = freeze_crossing(outcome)
    path = tmp_path / "airline.json"

    save_regression_probes([probe], path)
    loaded = load_regression_probes(path)

    assert loaded == [probe]
