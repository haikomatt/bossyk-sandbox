from __future__ import annotations

from collections.abc import Sequence

from bossyk_sandbox.gate import Gate
from bossyk_sandbox.instruments.base import (
    Decision,
    ObservedAction,
    ProposedAction,
    Verdict,
)
from bossyk_sandbox.instruments.hardcoded_rule import RequireLookupBeforeCancel


def test_cancel_without_lookup_is_blocked() -> None:
    gate = Gate(instruments=[RequireLookupBeforeCancel()])

    decision = gate.score(ProposedAction("cancel_reservation", {"reservation_id": "R1"}))

    assert decision.verdict is Verdict.BLOCK


def test_cancel_after_matching_lookup_is_allowed() -> None:
    gate = Gate(instruments=[RequireLookupBeforeCancel()])
    gate.record(ProposedAction("get_reservation_details", {"reservation_id": "R1"}))

    decision = gate.score(ProposedAction("cancel_reservation", {"reservation_id": "R1"}))

    assert decision.verdict is Verdict.ALLOW


def test_non_gated_tool_is_always_allowed() -> None:
    gate = Gate(instruments=[RequireLookupBeforeCancel()])

    decision = gate.score(ProposedAction("get_user_details", {"user_id": "U1"}))

    assert decision.verdict is Verdict.ALLOW


def test_lookup_for_different_reservation_does_not_authorise_cancel() -> None:
    gate = Gate(instruments=[RequireLookupBeforeCancel()])
    gate.record(ProposedAction("get_reservation_details", {"reservation_id": "R1"}))

    decision = gate.score(ProposedAction("cancel_reservation", {"reservation_id": "R2"}))

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


class _Fixed:
    """An instrument that returns one fixed verdict for every action."""

    def __init__(self, verdict: Verdict) -> None:
        self._decision = Decision(verdict, f"fixed: {verdict.value}")

    def score(
        self, proposed: ProposedAction, history: Sequence[ProposedAction | ObservedAction]
    ) -> Decision:
        return self._decision


def test_hold_is_returned_when_no_instrument_blocks() -> None:
    gate = Gate(instruments=[_Fixed(Verdict.ALLOW), _Fixed(Verdict.HOLD)])

    decision = gate.score(ProposedAction("bash", {"command": "rm -rf build"}))

    assert decision.verdict is Verdict.HOLD
    assert decision.reason == "fixed: hold"


def test_block_beats_hold_regardless_of_instrument_order() -> None:
    gate = Gate(instruments=[_Fixed(Verdict.HOLD), _Fixed(Verdict.BLOCK)])

    decision = gate.score(ProposedAction("bash", {"command": "curl x"}))

    assert decision.verdict is Verdict.BLOCK


def test_first_hold_wins_among_holds() -> None:
    first = _Fixed(Verdict.HOLD)
    first._decision = Decision(Verdict.HOLD, "first")
    gate = Gate(instruments=[first, _Fixed(Verdict.HOLD)])

    assert gate.score(ProposedAction("bash", {})).reason == "first"
