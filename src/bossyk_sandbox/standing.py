"""Standing / authority model (§F) — bossyk-proprietary enforcement.

Expresses what a session may do within bounds, so the enforcement layer can tell
a **within-authority** action (allow / defer) from an **over-authority** one
(escalate). This is the piece that makes `defer` derivable (see
`console/modes.py`): "within authority, async-approve" is fundamentally a
standing-model judgment the gate alone can't make.

v0 scope (decisions locked 2026-07-28): **count-only, session-scoped** — no
amounts, no wall-clock. Authority is "how many times this session may hit a
consequence boundary before it exceeds standing." Amount ceilings and real time
windows are deliberate follow-ups (need an amount/outcome oracle and a clock;
see the plan). Kept proprietary, not proposed to auditk-spec (per demonstrator
§13); the shape is a v0.2 candidate if a second implementer appears.
"""

from __future__ import annotations

from dataclasses import dataclass

from bossyk_sandbox.instruments.base import ProposedAction

# Each governed write tool maps to a consequence boundary (mirrors the demo
# taxonomy / compliance.attribution). Read tools are not governed here.
_TOOL_BOUNDARY: dict[str, str] = {
    "cancel_pending_order": "cancellation",
    "return_delivered_order_items": "refund",
    "modify_pending_order_payment": "payment_change",
    "modify_user_address": "account_change",
}


@dataclass(frozen=True)
class StandingGrant:
    """A session-scoped, count-only authority grant for one consequence
    boundary. `max_count` is how many times the boundary's action may be taken
    in the session before it exceeds standing authority."""

    boundary: str
    max_count: int


@dataclass(frozen=True)
class AuthorityVerdict:
    """`within` / `over` standing authority, or `not_governed` (no boundary or
    no grant). `boundary` is set whenever the action mapped to a boundary."""

    status: str
    reason: str
    boundary: str | None = None


def retail_standing_grants() -> dict[str, StandingGrant]:
    """The committed per-domain (retail) standing policy (decision #2). A demo
    default, not a real delegated-authority config: a session may cancel a
    couple of orders and issue one refund within authority; a further cancel is
    deferred (reversible), a further refund escalates (irreversible), and
    payment changes have no standing at all (any one escalates)."""
    return {
        "cancellation": StandingGrant(boundary="cancellation", max_count=2),
        "refund": StandingGrant(boundary="refund", max_count=1),
        "account_change": StandingGrant(boundary="account_change", max_count=1),
        "payment_change": StandingGrant(boundary="payment_change", max_count=0),
    }


def boundary_for(tool_name: str) -> str | None:
    """The consequence boundary a tool acts on, or None if it is not a governed
    write (e.g. a read/lookup)."""
    return _TOOL_BOUNDARY.get(tool_name)


def evaluate_authority(
    proposed: ProposedAction,
    history: list[ProposedAction],
    grants: dict[str, StandingGrant],
) -> AuthorityVerdict:
    """Count-only, session-scoped authority for `proposed`.

    Counts prior actions in `history` that hit the same boundary; the proposed
    action is the (count+1)-th. If that exceeds the boundary's grant it is
    `over`, else `within`. A tool with no boundary, or a boundary with no grant,
    is `not_governed` (the caller falls back to its non-authority heuristic).
    `history` is whatever the caller deems authority-consuming (e.g. the
    session's allowed actions).
    """
    boundary = boundary_for(proposed.tool_name)
    if boundary is None:
        return AuthorityVerdict("not_governed", f"{proposed.tool_name} is not a governed boundary")

    grant = grants.get(boundary)
    if grant is None:
        return AuthorityVerdict("not_governed", f"no standing grant for {boundary}", boundary)

    nth = sum(1 for action in history if boundary_for(action.tool_name) == boundary) + 1
    if nth > grant.max_count:
        return AuthorityVerdict(
            "over",
            f"{boundary}: action #{nth} exceeds standing max_count {grant.max_count}",
            boundary,
        )
    return AuthorityVerdict(
        "within", f"{boundary}: action #{nth} within standing max_count {grant.max_count}", boundary
    )
