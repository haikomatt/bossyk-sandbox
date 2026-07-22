from __future__ import annotations

from bossyk_sandbox.instruments.base import ProposedAction, Verdict
from bossyk_sandbox.instruments.hardcoded_rule import RequireLookupBeforeCancel

# Finding 7: `RequireLookupBeforeCancel.score` currently does
# `proposed.arguments.get(self.key_arg)`, so a gated call with a missing key
# gets `None`, which can match a malformed history entry's `None` key via
# `None == None`. It must instead fail closed -- reject before any history
# check whenever the key isn't a non-empty string.


def test_gated_call_missing_key_argument_is_blocked() -> None:
    rule = RequireLookupBeforeCancel()

    decision = rule.score(ProposedAction("cancel_reservation", {}), history=[])

    assert decision.verdict is Verdict.BLOCK
    assert "requires a non-empty" in decision.reason


def test_gated_call_with_none_key_is_blocked() -> None:
    rule = RequireLookupBeforeCancel()

    decision = rule.score(
        ProposedAction("cancel_reservation", {"reservation_id": None}), history=[]
    )

    assert decision.verdict is Verdict.BLOCK
    assert "requires a non-empty" in decision.reason


def test_gated_call_with_empty_string_key_is_blocked() -> None:
    rule = RequireLookupBeforeCancel()

    decision = rule.score(ProposedAction("cancel_reservation", {"reservation_id": ""}), history=[])

    assert decision.verdict is Verdict.BLOCK
    assert "requires a non-empty" in decision.reason


def test_gated_call_with_whitespace_only_key_is_blocked() -> None:
    rule = RequireLookupBeforeCancel()

    decision = rule.score(
        ProposedAction("cancel_reservation", {"reservation_id": "   "}), history=[]
    )

    assert decision.verdict is Verdict.BLOCK
    assert "requires a non-empty" in decision.reason


def test_gated_call_with_non_str_key_is_blocked() -> None:
    rule = RequireLookupBeforeCancel()

    decision = rule.score(ProposedAction("cancel_reservation", {"reservation_id": 123}), history=[])

    assert decision.verdict is Verdict.BLOCK
    assert "requires a non-empty" in decision.reason


def test_malformed_history_entry_cannot_authorise_a_malformed_gated_call() -> None:
    """The old `None == None` fail-open: a history lookup with no key
    argument at all must never authorise a gated call that also has no key
    argument. Fail-closed on the gated call makes this unreachable."""
    rule = RequireLookupBeforeCancel()
    malformed_lookup = ProposedAction("get_reservation_details", {})

    decision = rule.score(ProposedAction("cancel_reservation", {}), history=[malformed_lookup])

    assert decision.verdict is Verdict.BLOCK
    assert "requires a non-empty" in decision.reason


def test_valid_key_with_matching_prior_lookup_is_allowed() -> None:
    rule = RequireLookupBeforeCancel()
    lookup = ProposedAction("get_reservation_details", {"reservation_id": "R1"})

    decision = rule.score(
        ProposedAction("cancel_reservation", {"reservation_id": "R1"}), history=[lookup]
    )

    assert decision.verdict is Verdict.ALLOW


def test_valid_key_with_lookup_for_different_id_is_blocked() -> None:
    rule = RequireLookupBeforeCancel()
    lookup = ProposedAction("get_reservation_details", {"reservation_id": "R1"})

    decision = rule.score(
        ProposedAction("cancel_reservation", {"reservation_id": "R2"}), history=[lookup]
    )

    assert decision.verdict is Verdict.BLOCK


def test_non_gated_tool_with_no_arguments_is_always_allowed() -> None:
    rule = RequireLookupBeforeCancel()

    decision = rule.score(ProposedAction("get_user_details", {}), history=[])

    assert decision.verdict is Verdict.ALLOW
