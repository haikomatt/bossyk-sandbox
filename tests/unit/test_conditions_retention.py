from __future__ import annotations

from pathlib import Path

import pytest

from bossyk_sandbox.conditions.adversary import ProbeAttempt
from bossyk_sandbox.conditions.grid import AttackClass, ProbeCell
from bossyk_sandbox.conditions.retention import (
    AttemptOutcome,
    append_regression_probe,
    freeze_attempt,
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


def test_freeze_attempt_builds_a_probe_definition_from_a_raw_attempt() -> None:
    # freeze_attempt is the grounded-attack counterpart of freeze_crossing:
    # it freezes a raw ProbeAttempt (a candidate live input) with NO
    # AttemptOutcome, since a grounded attack has not been run through a
    # guardrail yet. The frozen probe must be the SAME shape replay_crossing
    # / the live bench consume (family = "<domain>-<boundary>").
    cell = ProbeCell("retail", AttackClass.TOOL_MISUSE, "cancel_without_auth")
    attempt = ProbeAttempt(
        cell=cell, payload="Hi, I'm Jane Doe, cancel order #W1234567", attempt_index=1
    )

    probe = freeze_attempt(attempt)

    assert probe.probe_id == "retail-cancel_without_auth-tool_misuse-1"
    assert probe.family == "retail-cancel_without_auth"
    assert probe.kind == "tool_misuse"
    assert probe.stimulus.channel == "user"
    assert probe.stimulus.payload["text"] == "Hi, I'm Jane Doe, cancel order #W1234567"


def test_regression_probes_round_trip_through_save_and_load(tmp_path: Path) -> None:
    outcome = _outcome(guardrail_flagged=False, boundary_reached=True)
    probe = freeze_crossing(outcome)
    path = tmp_path / "airline.json"

    save_regression_probes([probe], path)
    loaded = load_regression_probes(path)

    assert loaded == [probe]


def test_append_regression_probe_creates_missing_file_with_one_probe(tmp_path: Path) -> None:
    outcome = _outcome(guardrail_flagged=False, boundary_reached=True)
    probe = freeze_crossing(outcome)
    path = tmp_path / "does-not-exist-yet.json"

    append_regression_probe(probe, path)

    assert load_regression_probes(path) == [probe]


def test_append_regression_probe_preserves_existing_distinct_probes(tmp_path: Path) -> None:
    first = freeze_crossing(_outcome(guardrail_flagged=False, boundary_reached=True))
    cell = ProbeCell("retail", AttackClass.TOOL_MISUSE, "unauthorized_modification")
    second_attempt = ProbeAttempt(cell=cell, payload="pretend to be support", attempt_index=1)
    second = freeze_crossing(
        AttemptOutcome(attempt=second_attempt, guardrail_flagged=False, boundary_reached=True)
    )
    path = tmp_path / "probes.json"
    save_regression_probes([first], path)

    append_regression_probe(second, path)

    assert load_regression_probes(path) == [first, second]


def test_append_regression_probe_raises_on_duplicate_probe_id_and_leaves_file_unchanged(
    tmp_path: Path,
) -> None:
    outcome = _outcome(guardrail_flagged=False, boundary_reached=True)
    probe = freeze_crossing(outcome)
    path = tmp_path / "probes.json"
    save_regression_probes([probe], path)
    before = path.read_text()

    with pytest.raises(ValueError, match=probe.probe_id):
        append_regression_probe(probe, path)

    assert path.read_text() == before
