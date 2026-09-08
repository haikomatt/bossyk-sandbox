"""RED-phase contract: the tool -> boundary map becomes tool -> SET of boundaries.

`standing._TOOL_BOUNDARY` was 1:1 (tool -> single boundary) and its own
comment block recorded the failure: `place_call` / `send_sms` / `send_email`
each cross EITHER `contact_without_consent` OR `out_of_hours_contact`
depending on context, so `out_of_hours_contact` was unreachable through the
map. This module pins the replacement shape (`boundaries_for_tool`) and the
enforcement consequence: `evaluate_authority` is `over` when ANY of a tool's
boundaries exceeds its grant, and a prior multi-boundary action consumes
authority on EACH of its boundaries.
"""

from __future__ import annotations

from bossyk_sandbox.instruments.base import ProposedAction
from bossyk_sandbox.standing import StandingGrant, boundaries_for_tool, evaluate_authority

CONTACT_TOOLS = ("place_call", "send_sms", "send_email")


def _place_call(phone: str = "+447700900001") -> ProposedAction:
    return ProposedAction("place_call", {"phone": phone})


def _grant(boundary: str, max_count: int) -> StandingGrant:
    return StandingGrant(boundary=boundary, max_count=max_count)


# --- the map ------------------------------------------------------------------


def test_place_call_carries_both_contact_boundaries() -> None:
    assert boundaries_for_tool("place_call") == frozenset(
        {"contact_without_consent", "out_of_hours_contact"}
    )


def test_every_contact_tool_reaches_out_of_hours_contact() -> None:
    # Exit criterion: `out_of_hours_contact` is reachable via the map.
    for tool in CONTACT_TOOLS:
        assert "out_of_hours_contact" in boundaries_for_tool(tool), tool
        assert "contact_without_consent" in boundaries_for_tool(tool), tool


def test_single_boundary_tools_return_a_one_element_set() -> None:
    assert boundaries_for_tool("cancel_pending_order") == frozenset({"cancellation"})
    assert boundaries_for_tool("return_delivered_order_items") == frozenset({"refund"})
    assert boundaries_for_tool("modify_pending_order_payment") == frozenset({"payment_change"})
    assert boundaries_for_tool("modify_user_address") == frozenset({"account_change"})
    assert boundaries_for_tool("book_survey") == frozenset({"booking_without_eligibility"})
    assert boundaries_for_tool("apply_discount") == frozenset({"unauthorised_incentive"})


def test_ungoverned_read_tool_returns_an_empty_set() -> None:
    assert boundaries_for_tool("get_order_details") == frozenset()
    assert boundaries_for_tool("check_suppression") == frozenset()
    # `record_consent` is an audit write, not a boundary (spec).
    assert boundaries_for_tool("record_consent") == frozenset()


# --- enforcement over a multi-boundary tool ----------------------------------


def test_over_when_any_boundary_of_a_multi_boundary_tool_exceeds_its_grant() -> None:
    # Consent standing is generous; out-of-hours standing is zero. The first
    # call is within the consent grant but over the out-of-hours grant, so the
    # verdict is `over` and names the boundary that tripped.
    grants = {
        "contact_without_consent": _grant("contact_without_consent", max_count=5),
        "out_of_hours_contact": _grant("out_of_hours_contact", max_count=0),
    }
    verdict = evaluate_authority(_place_call(), [], grants)
    assert verdict.status == "over"
    assert verdict.boundary == "out_of_hours_contact"
    assert "out_of_hours_contact" in verdict.reason


def test_within_when_every_boundary_of_a_multi_boundary_tool_is_within() -> None:
    grants = {
        "contact_without_consent": _grant("contact_without_consent", max_count=5),
        "out_of_hours_contact": _grant("out_of_hours_contact", max_count=1),
    }
    assert evaluate_authority(_place_call(), [], grants).status == "within"


def test_prior_multi_boundary_action_consumes_authority_on_each_of_its_boundaries() -> None:
    # Only the out-of-hours boundary is granted (max 1). A prior place_call
    # must count against it even though the tool ALSO maps to consent --
    # the history filter has to match on set membership, not equality.
    grants = {"out_of_hours_contact": _grant("out_of_hours_contact", max_count=1)}
    verdict = evaluate_authority(_place_call(), [_place_call()], grants)
    assert verdict.status == "over"
    assert verdict.boundary == "out_of_hours_contact"


def test_multi_boundary_tool_with_a_grant_on_only_one_boundary_is_governed_by_it() -> None:
    # No out-of-hours grant at all: that boundary is ungoverned and must not
    # block, but the consent grant (max 0) still governs the tool.
    grants = {"contact_without_consent": _grant("contact_without_consent", max_count=0)}
    verdict = evaluate_authority(_place_call(), [], grants)
    assert verdict.status == "over"
    assert verdict.boundary == "contact_without_consent"


def test_multi_boundary_tool_with_no_grant_on_any_boundary_is_not_governed() -> None:
    grants = {"cancellation": _grant("cancellation", max_count=2)}
    verdict = evaluate_authority(_place_call(), [], grants)
    assert verdict.status == "not_governed"
