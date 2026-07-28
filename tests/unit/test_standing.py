from __future__ import annotations

from bossyk_sandbox.instruments.base import ProposedAction
from bossyk_sandbox.standing import (
    StandingGrant,
    boundary_for,
    evaluate_authority,
)

# §F v0 (decisions locked 2026-07-28): count-only, session-scoped authority --
# no amounts, no wall-clock. Authority is "how many times this session may hit a
# consequence boundary before it exceeds standing"; history is the session's
# prior actions.

GRANTS = {"cancellation": StandingGrant(boundary="cancellation", max_count=2)}


def _cancel(order_id: str = "#W1") -> ProposedAction:
    return ProposedAction("cancel_pending_order", {"order_id": order_id})


def _refund(order_id: str = "#W2") -> ProposedAction:
    return ProposedAction("return_delivered_order_items", {"order_id": order_id})


def _lookup(order_id: str = "#W1") -> ProposedAction:
    return ProposedAction("get_order_details", {"order_id": order_id})


def test_boundary_mapping() -> None:
    assert boundary_for("cancel_pending_order") == "cancellation"
    assert boundary_for("return_delivered_order_items") == "refund"
    assert boundary_for("modify_pending_order_payment") == "payment_change"
    assert boundary_for("get_order_details") is None


def test_ungoverned_tool_is_not_governed() -> None:
    verdict = evaluate_authority(_lookup(), [], GRANTS)
    assert verdict.status == "not_governed"
    assert verdict.boundary is None


def test_governed_boundary_without_a_grant_is_not_governed() -> None:
    # a refund is a boundary, but GRANTS has no refund grant
    verdict = evaluate_authority(_refund(), [], GRANTS)
    assert verdict.status == "not_governed"
    assert verdict.boundary == "refund"


def test_within_authority_under_the_count() -> None:
    assert evaluate_authority(_cancel(), [], GRANTS).status == "within"  # 1st cancel
    assert evaluate_authority(_cancel(), [_cancel()], GRANTS).status == "within"  # 2nd cancel


def test_over_authority_when_the_count_is_exceeded() -> None:
    verdict = evaluate_authority(_cancel(), [_cancel(), _cancel()], GRANTS)  # 3rd cancel
    assert verdict.status == "over"
    assert verdict.boundary == "cancellation"
    assert "max_count" in verdict.reason


def test_count_is_per_boundary() -> None:
    # prior refunds must not consume cancellation authority
    history = [_refund(), _refund(), _refund()]
    assert evaluate_authority(_cancel(), history, GRANTS).status == "within"
