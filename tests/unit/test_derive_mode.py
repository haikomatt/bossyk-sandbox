from __future__ import annotations

from bossyk_sandbox.console.modes import derive_mode
from bossyk_sandbox.instruments.base import Verdict
from bossyk_sandbox.standing import AuthorityVerdict

# derive_mode maps a REAL gate verdict + the tool's consequence class onto a
# resolution mode -- the honest upgrade over Slice A's authored per-turn modes.
# The consequence table is a documented stand-in for the standing/authority
# model (§F, deferred); until that lands, `defer` (within-authority async
# approval) is NOT derivable and is deliberately absent from live derivation.


def test_blocked_reversible_write_redirects() -> None:
    d = derive_mode("cancel_pending_order", Verdict.BLOCK)
    assert d.mode == "redirect"
    assert d.hitl is None


def test_blocked_irreversible_write_escalates_with_its_severity() -> None:
    payment = derive_mode("modify_pending_order_payment", Verdict.BLOCK)
    assert payment.mode == "escalate"
    assert payment.hitl is not None and payment.hitl["severity"] == "critical"

    refund = derive_mode("return_delivered_order_items", Verdict.BLOCK)
    assert refund.mode == "escalate"
    assert refund.hitl is not None and refund.hitl["severity"] == "high"
    assert refund.hitl["reason"] and refund.hitl["resolution"]


def test_allowed_pii_read_steps_up() -> None:
    assert derive_mode("get_user_details", Verdict.ALLOW).mode == "step-up"
    assert derive_mode("find_user_id", Verdict.ALLOW).mode == "step-up"


def test_allowed_benign_read_allows() -> None:
    d = derive_mode("get_order_details", Verdict.ALLOW)
    assert d.mode == "allow"
    assert d.hitl is None


def test_allowed_write_with_prior_lookup_allows() -> None:
    # A looked-up (gate-ALLOWed) reversible write just passes -- `defer` would
    # require the standing model to say it's within authority.
    assert derive_mode("cancel_pending_order", Verdict.ALLOW).mode == "allow"


def test_unknown_tool_is_conservative() -> None:
    # Unknown tool: treat as an irreversible mutation -> a blocked one escalates
    # (to a human) rather than being silently redirected/allowed.
    assert derive_mode("some_unknown_tool", Verdict.ALLOW).mode == "allow"
    blocked = derive_mode("some_unknown_tool", Verdict.BLOCK)
    assert blocked.mode == "escalate"


def test_defer_is_not_derivable_without_the_standing_model() -> None:
    # No (tool, verdict) pair yields `defer` from the consequence table alone.
    tools = [
        "get_order_details",
        "get_user_details",
        "find_user_id",
        "cancel_pending_order",
        "return_delivered_order_items",
        "modify_pending_order_payment",
        "modify_user_address",
        "some_unknown_tool",
    ]
    modes = {derive_mode(t, v).mode for t in tools for v in (Verdict.ALLOW, Verdict.BLOCK)}
    assert "defer" not in modes
    assert modes <= {"allow", "redirect", "step-up", "escalate"}


# --- §F: authority-driven ALLOW-path modes -----------------------------------


def _over(boundary: str = "cancellation") -> AuthorityVerdict:
    return AuthorityVerdict("over", "test: over standing", boundary)


def _within(boundary: str = "cancellation") -> AuthorityVerdict:
    return AuthorityVerdict("within", "test: within standing", boundary)


def test_over_authority_reversible_now_defers() -> None:
    # This is the whole point of §F: `defer` becomes derivable.
    d = derive_mode("cancel_pending_order", Verdict.ALLOW, authority=_over())
    assert d.mode == "defer"
    assert d.hitl is None


def test_over_authority_irreversible_escalates() -> None:
    refund = derive_mode("return_delivered_order_items", Verdict.ALLOW, authority=_over("refund"))
    assert refund.mode == "escalate"
    assert refund.hitl is not None and refund.hitl["severity"] == "high"

    payment = derive_mode(
        "modify_pending_order_payment", Verdict.ALLOW, authority=_over("payment_change")
    )
    assert payment.mode == "escalate"
    assert payment.hitl is not None and payment.hitl["severity"] == "critical"


def test_within_authority_allows_as_before() -> None:
    assert derive_mode("cancel_pending_order", Verdict.ALLOW, authority=_within()).mode == "allow"


def test_within_authority_pii_still_steps_up() -> None:
    d = derive_mode("modify_user_address", Verdict.ALLOW, authority=_within("account_change"))
    assert d.mode == "step-up"


def test_not_governed_authority_is_back_compatible() -> None:
    not_gov = AuthorityVerdict("not_governed", "no grant")
    assert derive_mode("cancel_pending_order", Verdict.ALLOW, authority=not_gov).mode == "allow"
