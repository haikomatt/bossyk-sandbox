from __future__ import annotations

from bossyk_sandbox.gate import Gate
from bossyk_sandbox.instruments.base import ProposedAction, Verdict
from bossyk_sandbox.instruments.hardcoded_rule import RequireLookupBeforeCancel


def test_cancel_without_lookup_is_blocked() -> None:
    gate = Gate(instruments=[RequireLookupBeforeCancel()])

    decision = gate.evaluate(ProposedAction("cancel_reservation", {"reservation_id": "R1"}))

    assert decision.verdict is Verdict.BLOCK


def test_cancel_after_matching_lookup_is_allowed() -> None:
    gate = Gate(instruments=[RequireLookupBeforeCancel()])
    gate.evaluate(ProposedAction("get_reservation_details", {"reservation_id": "R1"}))

    decision = gate.evaluate(ProposedAction("cancel_reservation", {"reservation_id": "R1"}))

    assert decision.verdict is Verdict.ALLOW


def test_non_gated_tool_is_always_allowed() -> None:
    gate = Gate(instruments=[RequireLookupBeforeCancel()])

    decision = gate.evaluate(ProposedAction("get_user_details", {"user_id": "U1"}))

    assert decision.verdict is Verdict.ALLOW


def test_lookup_for_different_reservation_does_not_authorise_cancel() -> None:
    gate = Gate(instruments=[RequireLookupBeforeCancel()])
    gate.evaluate(ProposedAction("get_reservation_details", {"reservation_id": "R1"}))

    decision = gate.evaluate(ProposedAction("cancel_reservation", {"reservation_id": "R2"}))

    assert decision.verdict is Verdict.BLOCK


def test_score_does_not_mutate_history() -> None:
    gate = Gate(instruments=[RequireLookupBeforeCancel()])
    lookup = ProposedAction("get_reservation_details", {"reservation_id": "R1"})

    gate.score(lookup)
    gate.score(lookup)
    decision = gate.score(ProposedAction("cancel_reservation", {"reservation_id": "R1"}))

    assert decision.verdict is Verdict.BLOCK


def test_history_property_reflects_recorded_calls() -> None:
    gate = Gate(instruments=[RequireLookupBeforeCancel()])
    lookup = ProposedAction("get_reservation_details", {"reservation_id": "R1"})
    gate.record(lookup)

    history = gate.history
    assert history == [lookup]

    history.append(ProposedAction("cancel_reservation", {"reservation_id": "R1"}))
    assert gate.history == [lookup]
