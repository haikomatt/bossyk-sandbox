"""Derive a resolution mode from a REAL gate verdict + the tool's consequence.

This is the Slice B upgrade over Slice A's authored per-turn modes: the mode is
computed from what actually happened at the gate plus a per-tool consequence
classification, not hand-written into a preset.

The consequence table is a documented stand-in for the standing / authority
model (§F of the demonstrator plan, deferred). Until that lands, `defer`
(within-authority async approval) is NOT derivable -- the gate alone cannot tell
a within-limit refund (defer) from an over-limit one (escalate) -- so live
derivation yields only allow / redirect / step-up / escalate. `defer` returns
when the standing model does. See `enforcement-delivery-model-v0.1` (vault).
"""

from __future__ import annotations

from dataclasses import dataclass

from bossyk_sandbox.instruments.base import Verdict
from bossyk_sandbox.standing import AuthorityVerdict

# HITL queue ordering, most severe first.
SEVERITY_RANK = {"critical": 0, "high": 1, "medium": 2, "low": 3}


@dataclass(frozen=True)
class Consequence:
    """What a tool does, structurally -- a defensible fact about the tool, not
    a content heuristic on its arguments (mirrors compliance.attribution's
    Tier-C stance). `severity` is used only when a blocked irreversible action
    escalates."""

    mutation: bool
    reversible: bool
    pii: bool
    severity: str


# Keyed by the tau2 retail tool's structural purpose. Unknown tools fall back
# to `_DEFAULT` -- treated as an irreversible mutation so a blocked unknown
# escalates to a human rather than being silently redirected or allowed.
_CONSEQUENCE: dict[str, Consequence] = {
    "get_order_details": Consequence(mutation=False, reversible=True, pii=False, severity="low"),
    "get_user_details": Consequence(mutation=False, reversible=True, pii=True, severity="low"),
    "find_user_id": Consequence(mutation=False, reversible=True, pii=True, severity="low"),
    "cancel_pending_order": Consequence(mutation=True, reversible=True, pii=False, severity="high"),
    "modify_user_address": Consequence(mutation=True, reversible=True, pii=True, severity="high"),
    "return_delivered_order_items": Consequence(
        mutation=True, reversible=False, pii=False, severity="high"
    ),
    "modify_pending_order_payment": Consequence(
        mutation=True, reversible=False, pii=False, severity="critical"
    ),
}
_DEFAULT = Consequence(mutation=True, reversible=False, pii=False, severity="medium")


@dataclass(frozen=True)
class ModeDecision:
    """A derived resolution mode plus, for `escalate`, the hard-cell HITL item."""

    mode: str
    reason: str
    hitl: dict[str, str] | None = None


def _escalation(
    tool_name: str, consequence: Consequence, *, mode_reason: str, hitl_reason: str
) -> ModeDecision:
    """An escalate decision with its hard-cell HITL item (by tool severity)."""
    return ModeDecision(
        mode="escalate",
        reason=mode_reason,
        hitl={
            "severity": consequence.severity,
            "reason": hitl_reason,
            "resolution": "queued for supervisor approval",
            "channel": "callback",
        },
    )


def derive_mode(
    tool_name: str, gate_verdict: Verdict, *, authority: AuthorityVerdict | None = None
) -> ModeDecision:
    """Map (gate verdict, tool consequence, standing authority) onto a mode.

    - BLOCK + reversible write               -> redirect  (agent re-runs the lookup)
    - BLOCK + irreversible write             -> escalate  (hard-cell HITL)
    - ALLOW + over authority + reversible    -> defer     (accept, async-approve)
    - ALLOW + over authority + irreversible  -> escalate  (a human must decide)
    - ALLOW + within authority + PII         -> step-up
    - ALLOW + within authority / not-governed -> allow

    `authority` is the §F standing verdict (`bossyk_sandbox.standing`). Without it
    -- or when `not_governed` -- the ALLOW path keeps its pre-§F behaviour and
    never yields `defer`: `defer` requires the standing model to say an action is
    *over* authority yet *reversible* (accept it, approve out of band). Over +
    irreversible goes to a human.
    """
    consequence = _CONSEQUENCE.get(tool_name, _DEFAULT)

    if gate_verdict is Verdict.BLOCK:
        if consequence.reversible:
            return ModeDecision(
                mode="redirect",
                reason="gate blocked an unverified reversible write; agent redirected to verify",
            )
        return _escalation(
            tool_name,
            consequence,
            mode_reason="irreversible action over delegated authority, escalated to human review",
            hitl_reason=(
                f"{tool_name}: irreversible action blocked pre-execution and exceeds "
                "delegated authority -- needs supervisor review."
            ),
        )

    # ALLOW path: the gate permitted it; standing authority decides the rest.
    if authority is not None and authority.status == "over":
        if consequence.reversible:
            return ModeDecision(
                mode="defer",
                reason="over standing authority but reversible -- accept + async-approve",
            )
        return _escalation(
            tool_name,
            consequence,
            mode_reason="over standing authority and irreversible -- escalated to review",
            hitl_reason=f"{tool_name}: over standing authority and irreversible -- needs review.",
        )

    if consequence.pii:
        return ModeDecision(
            mode="step-up",
            reason="personal-data access -- customer re-authentication required (not a reviewer)",
        )
    return ModeDecision(mode="allow", reason="passed the gate")
