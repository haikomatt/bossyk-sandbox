"""RED-phase contract: composition closure (co-occurrence restrictions) in the gate.

A restriction set `X` over action classes (standing boundaries) prohibits two
classes from BOTH occurring in one session, even when each action alone is
within standing. This is the Bounded Agents `X subset (A choose 2)` idea
wired into `evaluate_authority`, reading the `history` it already takes.

The retail seed is NOT hand-authored: `co_occurrence_restrictions` derives
it mechanically from `live_boundary.py`'s STRUCTURAL specs -- two structural
specs whose action tools mutate the same entity type (same `lookup_tool` +
`key_arg`) form a restricted pair, mapped through `boundaries_for_tool`.
Airline's tools carry no standing boundary, so its derived set is an honest
empty set, not a claim.
"""

from __future__ import annotations

from bossyk_sandbox.conditions.live_boundary import (
    AIRLINE_BOUNDARY_SPECS,
    OUTREACH_BOUNDARY_SPECS,
    RETAIL_BOUNDARY_SPECS,
    co_occurrence_restrictions,
)
from bossyk_sandbox.instruments.base import ProposedAction
from bossyk_sandbox.standing import (
    StandingGrant,
    TimedAction,
    evaluate_authority,
    retail_standing_restrictions,
)

# Generous grants: every action alone is comfortably within standing, so any
# `over` below can only come from the restriction set.
GRANTS = {
    "cancellation": StandingGrant(boundary="cancellation", max_count=5),
    "refund": StandingGrant(boundary="refund", max_count=5),
    "account_change": StandingGrant(boundary="account_change", max_count=5),
}
RESTRICTED = frozenset({("cancellation", "refund")})


def _cancel(order_id: str = "#W1") -> ProposedAction:
    return ProposedAction("cancel_pending_order", {"order_id": order_id})


def _refund(order_id: str = "#W1") -> ProposedAction:
    return ProposedAction("return_delivered_order_items", {"order_id": order_id})


def _address_change(user_id: str = "u1") -> ProposedAction:
    return ProposedAction("modify_user_address", {"user_id": user_id})


# --- the check ----------------------------------------------------------------


def test_cancel_then_refund_is_refused_when_the_pair_is_restricted() -> None:
    verdict = evaluate_authority(_refund(), [_cancel()], GRANTS, restrictions=RESTRICTED)
    assert verdict.status == "over"
    assert verdict.boundary == "refund"
    assert "cancellation" in verdict.reason
    assert "co-occurrence" in verdict.reason


def test_refund_then_cancel_is_refused_in_the_other_order() -> None:
    verdict = evaluate_authority(_cancel(), [_refund()], GRANTS, restrictions=RESTRICTED)
    assert verdict.status == "over"
    assert verdict.boundary == "cancellation"
    assert "refund" in verdict.reason


def test_each_action_alone_is_within_standing_under_the_same_restrictions() -> None:
    assert evaluate_authority(_cancel(), [], GRANTS, restrictions=RESTRICTED).status == "within"
    assert evaluate_authority(_refund(), [], GRANTS, restrictions=RESTRICTED).status == "within"


def test_an_unrestricted_pair_still_passes() -> None:
    verdict = evaluate_authority(_cancel(), [_address_change()], GRANTS, restrictions=RESTRICTED)
    assert verdict.status == "within"


def test_restriction_pairs_are_unordered() -> None:
    reversed_pair = frozenset({("refund", "cancellation")})
    refund_after_cancel = evaluate_authority(
        _refund(), [_cancel()], GRANTS, restrictions=reversed_pair
    )
    cancel_after_refund = evaluate_authority(
        _cancel(), [_refund()], GRANTS, restrictions=reversed_pair
    )
    assert refund_after_cancel.status == "over"
    assert cancel_after_refund.status == "over"


def test_restriction_applies_even_without_a_grant_on_the_proposed_boundary() -> None:
    # A restriction is an independent constraint, not a property of a grant:
    # with no refund grant at all, a refund after a cancel is still refused
    # rather than falling through to `not_governed`.
    cancel_only = {"cancellation": StandingGrant(boundary="cancellation", max_count=5)}
    verdict = evaluate_authority(_refund(), [_cancel()], cancel_only, restrictions=RESTRICTED)
    assert verdict.status == "over"


def test_timed_history_is_matched_too() -> None:
    history = [TimedAction(_cancel(), at=0.0)]
    verdict = evaluate_authority(_refund(), history, GRANTS, restrictions=RESTRICTED)
    assert verdict.status == "over"


def test_ungoverned_read_in_history_never_triggers_a_restriction() -> None:
    lookup = ProposedAction("get_order_details", {"order_id": "#W1"})
    verdict = evaluate_authority(_refund(), [lookup], GRANTS, restrictions=RESTRICTED)
    assert verdict.status == "within"


def test_no_restrictions_is_byte_identical_to_before() -> None:
    plain = evaluate_authority(_refund(), [_cancel()], GRANTS)
    empty = evaluate_authority(_refund(), [_cancel()], GRANTS, restrictions=frozenset())
    assert empty == plain
    assert plain.status == "within"


# --- seeding from live_boundary.py's structural pairs ------------------------


def test_retail_restrictions_are_seeded_from_the_structural_specs() -> None:
    # The committed retail set must equal the mechanical derivation -- it is
    # seeded, not authored. cancel_without_auth and refund_over_threshold
    # both mutate an order (get_order_details / order_id), so cancellation
    # and refund may not co-occur in one session (a cancel already refunds
    # everything paid; a return on top refunds the items again).
    assert retail_standing_restrictions() == frozenset({("cancellation", "refund")})
    assert retail_standing_restrictions() == co_occurrence_restrictions(RETAIL_BOUNDARY_SPECS)


def test_airline_derives_an_honest_empty_set() -> None:
    # cancel_reservation / update_reservation_flights share an entity but carry
    # no standing boundary, so nothing is enforceable -- and nothing is claimed.
    assert co_occurrence_restrictions(AIRLINE_BOUNDARY_SPECS) == frozenset()


def test_outreach_derives_an_empty_set_because_its_structural_pairs_differ() -> None:
    # place_call (phone / check_suppression) and book_survey (prospect_id /
    # check_eligibility) mutate different entities.
    assert co_occurrence_restrictions(OUTREACH_BOUNDARY_SPECS) == frozenset()
