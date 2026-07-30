"""RED-phase tests for slice-2 P5's history-widening compatibility
requirement (bossyk-sandbox scope-doc D3): existing instruments that only
read `.tool_name`/`.arguments` off history items must keep behaving
identically when history is `Sequence[ProposedAction | ObservedAction]`
instead of plain `list[ProposedAction]`, via the same `_action_of` unwrapper
`RequirePassedCheck` uses (see test_observed_action.py). This is the
load-bearing compatibility requirement: airline, retail, and outreach's
existing `RequireLookupBeforeCancel` fast rules must not need to change
their own logic, only unwrap.

Each assertion here mirrors an existing bare-ProposedAction-history behaviour
already locked in by tests/unit/test_hardcoded_rule.py -- this file only adds
the ObservedAction-wrapped-history side of the same cases.

NOTE for the RED gate: as written today (pre-GREEN), `RequireLookupBeforeCancel
.score` reads `call.tool_name` / `call.arguments` directly off each history
item without unwrapping -- so once `ObservedAction` exists (this file's
ImportError is resolved), these tests are expected to keep failing, now via
`AttributeError: 'ObservedAction' object has no attribute 'tool_name'`, until
GREEN adds the `_action_of` unwrap to `RequireLookupBeforeCancel.score` itself.
"""

from __future__ import annotations

from bossyk_sandbox.instruments.base import ObservedAction, ProposedAction, Verdict
from bossyk_sandbox.instruments.hardcoded_rule import RequireLookupBeforeCancel
from bossyk_sandbox.scenarios.runner import outreach_fast_rules, retail_fast_rules


def test_require_lookup_before_cancel_allows_with_an_observed_action_in_history() -> None:
    rule = RequireLookupBeforeCancel()
    lookup = ProposedAction("get_reservation_details", {"reservation_id": "R1"})
    observed_lookup = ObservedAction(action=lookup, result="some tool result")

    bare_decision = rule.score(
        ProposedAction("cancel_reservation", {"reservation_id": "R1"}), history=[lookup]
    )
    wrapped_decision = rule.score(
        ProposedAction("cancel_reservation", {"reservation_id": "R1"}), history=[observed_lookup]
    )

    assert bare_decision.verdict is Verdict.ALLOW
    assert wrapped_decision.verdict == bare_decision.verdict


def test_require_lookup_before_cancel_blocks_with_an_observed_action_for_a_different_id() -> None:
    rule = RequireLookupBeforeCancel()
    lookup = ProposedAction("get_reservation_details", {"reservation_id": "R1"})
    observed_lookup = ObservedAction(action=lookup, result="some tool result")

    decision = rule.score(
        ProposedAction("cancel_reservation", {"reservation_id": "R2"}), history=[observed_lookup]
    )

    assert decision.verdict is Verdict.BLOCK


def test_retail_fast_rule_allows_with_observed_action_history() -> None:
    rules = retail_fast_rules()
    cancel_rule = next(
        r
        for r in rules
        if isinstance(r, RequireLookupBeforeCancel) and r.gated_tool == "cancel_pending_order"
    )
    lookup = ProposedAction("get_order_details", {"order_id": "#W1"})
    observed_lookup = ObservedAction(action=lookup, result="ok")

    decision = cancel_rule.score(
        ProposedAction("cancel_pending_order", {"order_id": "#W1"}), history=[observed_lookup]
    )

    assert decision.verdict is Verdict.ALLOW


def test_outreach_fast_rule_allows_with_observed_action_history() -> None:
    rules = outreach_fast_rules()
    book_survey_rule = next(
        r
        for r in rules
        if isinstance(r, RequireLookupBeforeCancel) and r.gated_tool == "book_survey"
    )
    lookup = ProposedAction("check_eligibility", {"prospect_id": "P-1"})
    observed_lookup = ObservedAction(action=lookup, result={"eligible": True})

    decision = book_survey_rule.score(
        ProposedAction("book_survey", {"prospect_id": "P-1", "slot": "2026-08-03T10:00"}),
        history=[observed_lookup],
    )

    assert decision.verdict is Verdict.ALLOW


def test_require_lookup_before_cancel_still_blocks_with_no_matching_observed_action() -> None:
    rule = RequireLookupBeforeCancel()
    unrelated_observed = ObservedAction(
        action=ProposedAction("get_reservation_details", {"reservation_id": "OTHER"}),
        result="some tool result",
    )

    decision = rule.score(
        ProposedAction("cancel_reservation", {"reservation_id": "R1"}), history=[unrelated_observed]
    )

    assert decision.verdict is Verdict.BLOCK
