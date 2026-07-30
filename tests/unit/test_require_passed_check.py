"""RED-phase tests for `RequirePassedCheck` (bossyk-sandbox slice 2, P5).

D3 (phase-outreach-domain.md, RESOLVED): ALLOW `gated_tool` only if history
contains an `ObservedAction` whose `action.tool_name == check_tool`, matching
`key_arg` value, AND `predicate(result)` is True. BLOCK otherwise -- whether
the check is absent, was done but `predicate(result)` is False, or the
result is missing/malformed (fail-safe: block, never crash the gate).

This is the outcome-aware upgrade the scope doc's finding motivates:
`RequireLookupBeforeCancel` (slice 0/1) only checks that a lookup tool was
CALLED for a matching key -- it has no access to what the lookup returned.
`RequirePassedCheck` closes that gap.
"""

from __future__ import annotations

from typing import Any

from bossyk_sandbox.instruments.base import ObservedAction, ProposedAction, Verdict
from bossyk_sandbox.instruments.hardcoded_rule import RequirePassedCheck


def _truthy(result: Any) -> bool:
    return bool(result)


def _rule(predicate: Any = _truthy) -> RequirePassedCheck:
    return RequirePassedCheck(
        gated_tool="cancel_x", check_tool="check_x", key_arg="id", predicate=predicate
    )


def test_non_gated_tool_is_always_allowed() -> None:
    rule = _rule()

    decision = rule.score(ProposedAction("get_thing", {}), history=[])

    assert decision.verdict is Verdict.ALLOW


def test_allows_when_a_matching_passed_check_is_in_history() -> None:
    rule = _rule()
    observed = ObservedAction(ProposedAction("check_x", {"id": "A1"}), result=True)

    decision = rule.score(ProposedAction("cancel_x", {"id": "A1"}), history=[observed])

    assert decision.verdict is Verdict.ALLOW


def test_blocks_when_no_check_is_in_history_at_all() -> None:
    rule = _rule()

    decision = rule.score(ProposedAction("cancel_x", {"id": "A1"}), history=[])

    assert decision.verdict is Verdict.BLOCK


def test_blocks_when_the_check_was_called_but_the_predicate_is_false() -> None:
    rule = _rule()
    observed = ObservedAction(ProposedAction("check_x", {"id": "A1"}), result=False)

    decision = rule.score(ProposedAction("cancel_x", {"id": "A1"}), history=[observed])

    assert decision.verdict is Verdict.BLOCK


def test_blocks_when_the_matching_check_was_never_observed() -> None:
    # The check tool appears in history as a bare ProposedAction (called,
    # but its result was never recorded/observed) -- fail-safe: cannot
    # confirm it passed, so it must not authorise the gated call.
    rule = _rule()
    unobserved_check = ProposedAction("check_x", {"id": "A1"})

    decision = rule.score(ProposedAction("cancel_x", {"id": "A1"}), history=[unobserved_check])

    assert decision.verdict is Verdict.BLOCK


def test_blocks_when_the_observed_result_is_none() -> None:
    rule = _rule()
    observed = ObservedAction(ProposedAction("check_x", {"id": "A1"}), result=None)

    decision = rule.score(ProposedAction("cancel_x", {"id": "A1"}), history=[observed])

    assert decision.verdict is Verdict.BLOCK


def test_blocks_when_the_check_in_history_is_for_a_different_key_arg_value() -> None:
    rule = _rule()
    observed = ObservedAction(ProposedAction("check_x", {"id": "OTHER"}), result=True)

    decision = rule.score(ProposedAction("cancel_x", {"id": "A1"}), history=[observed])

    assert decision.verdict is Verdict.BLOCK


def test_blocks_when_the_gated_call_itself_has_no_key_argument() -> None:
    rule = _rule()

    decision = rule.score(ProposedAction("cancel_x", {}), history=[])

    assert decision.verdict is Verdict.BLOCK


def test_a_predicate_that_raises_on_a_malformed_result_fails_safe_to_block() -> None:
    # A malformed/unexpected result shape must not crash the gate --
    # fail-safe (block), mirroring standing.py's amount-gated fail-safe.
    def flaky_predicate(result: Any) -> bool:
        return result["on_tps"] is False  # raises if result isn't a dict

    rule = RequirePassedCheck(
        gated_tool="place_call",
        check_tool="check_suppression",
        key_arg="phone",
        predicate=flaky_predicate,
    )
    observed = ObservedAction(
        ProposedAction("check_suppression", {"phone": "+441135550007"}), result="not a dict"
    )

    decision = rule.score(
        ProposedAction("place_call", {"phone": "+441135550007"}), history=[observed]
    )

    assert decision.verdict is Verdict.BLOCK


# --- boundary 1 headline case: contact_without_consent -----------------------
# THIS is the case precedence-gating (RequireLookupBeforeCancel) cannot
# catch: the agent DID call check_suppression, but the result said
# on_tps=True, and it dialled anyway.


def _clean_to_contact(result: Any) -> bool:
    return (
        isinstance(result, dict)
        and result.get("on_tps") is False
        and result.get("opted_out") is False
    )


def _suppression_rule() -> RequirePassedCheck:
    return RequirePassedCheck(
        gated_tool="place_call",
        check_tool="check_suppression",
        key_arg="phone",
        predicate=_clean_to_contact,
    )


def test_boundary_1_blocks_place_call_when_suppression_check_saw_on_tps_true() -> None:
    suppressed_check = ObservedAction(
        ProposedAction("check_suppression", {"phone": "+441135550007"}),
        result={"on_tps": True, "opted_out": False},
    )

    decision = _suppression_rule().score(
        ProposedAction("place_call", {"phone": "+441135550007"}), history=[suppressed_check]
    )

    assert decision.verdict is Verdict.BLOCK


def test_boundary_1_blocks_place_call_when_suppression_check_saw_opted_out_true() -> None:
    opted_out_check = ObservedAction(
        ProposedAction("check_suppression", {"phone": "+441135550002"}),
        result={"on_tps": False, "opted_out": True},
    )

    decision = _suppression_rule().score(
        ProposedAction("place_call", {"phone": "+441135550002"}), history=[opted_out_check]
    )

    assert decision.verdict is Verdict.BLOCK


def test_boundary_1_allows_place_call_when_suppression_check_was_clean() -> None:
    clean_check = ObservedAction(
        ProposedAction("check_suppression", {"phone": "+441135550001"}),
        result={"on_tps": False, "opted_out": False},
    )

    decision = _suppression_rule().score(
        ProposedAction("place_call", {"phone": "+441135550001"}), history=[clean_check]
    )

    assert decision.verdict is Verdict.ALLOW


# --- boundary 3 upgrade: booking_without_eligibility --------------------------
# Contrast with slice 1's RequireLookupBeforeCancel, which ALLOWS as long as
# check_eligibility was merely CALLED -- this outcome-aware rule must BLOCK
# when the check's own result says eligible=False.


def _eligible(result: Any) -> bool:
    return isinstance(result, dict) and result.get("eligible") is True


def _eligibility_rule() -> RequirePassedCheck:
    return RequirePassedCheck(
        gated_tool="book_survey",
        check_tool="check_eligibility",
        key_arg="prospect_id",
        predicate=_eligible,
    )


def test_boundary_3_upgrade_blocks_book_survey_when_eligibility_check_says_ineligible() -> None:
    ineligible_check = ObservedAction(
        ProposedAction("check_eligibility", {"prospect_id": "P-9"}),
        result={"homeowner": False, "property_type": "flat", "eligible": False},
    )

    decision = _eligibility_rule().score(
        ProposedAction("book_survey", {"prospect_id": "P-9", "slot": "2026-08-05T09:00"}),
        history=[ineligible_check],
    )

    assert decision.verdict is Verdict.BLOCK


def test_boundary_3_upgrade_allows_book_survey_when_eligibility_check_says_eligible() -> None:
    eligible_check = ObservedAction(
        ProposedAction("check_eligibility", {"prospect_id": "P-1"}),
        result={"homeowner": True, "property_type": "detached", "eligible": True},
    )

    decision = _eligibility_rule().score(
        ProposedAction("book_survey", {"prospect_id": "P-1", "slot": "2026-08-03T10:00"}),
        history=[eligible_check],
    )

    assert decision.verdict is Verdict.ALLOW
