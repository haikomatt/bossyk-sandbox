"""Standing / authority model (§F) — bossyk-proprietary enforcement.

Expresses what a session may do within bounds, so the enforcement layer can tell
a **within-authority** action (allow / defer) from an **over-authority** one
(escalate). This is the piece that makes `defer` derivable (see
`console/modes.py`): "within authority, async-approve" is fundamentally a
standing-model judgment the gate alone can't make.

v0 scope: **count-only** authority — "how many times a session may hit a
consequence boundary before it exceeds standing." A grant is session-scoped by
default; giving it a `window` (e.g. "1h") makes standing **time-bounded** so
authority *expires* — a prior action older than the window no longer consumes
it. `now` is always injected (never read from a real clock in this module) so
evaluation stays pure and deterministic.

A grant may ALSO carry `max_amount`: cumulative-budget authority -- "how much
value a session may move at a consequence boundary before it exceeds
standing," gating by *value* rather than just count. Count is the unit-1
special case of the same idea: sum the in-window consumed amounts (stamped on
prior `TimedAction`s, exactly like `at`) plus the proposed action's resolved
`amount`, and it is `over` if that exceeds `max_amount`. A grant with both
`max_count` and `max_amount` is `over` if EITHER is exceeded. The amount
itself is resolved upstream by the oracle (`standing_amount.resolve_amount`)
and injected here, same pattern as `now` for windows -- this module never
looks anything up itself. An unresolvable amount (`amount=None`) on an
amount-gated grant is always `over`: it can never prove it's within budget, so
it fails safe rather than silently allowing. `max_amount=None` (the default)
keeps a grant exactly as before -- not amount-gated at all. Kept proprietary,
not proposed to auditk-spec (per demonstrator §13); the shape is a v0.2
candidate if a second implementer appears.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from bossyk_sandbox.instruments.base import ProposedAction

# Each governed write tool maps to a consequence boundary (mirrors the demo
# taxonomy / compliance.attribution). Read tools are not governed here.
#
# D5 (kept as a single global, domain-agnostic map, not a per-domain
# registry): `boundary_for` is called without a domain everywhere it's used
# (this module, standing_amount.py, live_boundary.py) -- promoting to a
# per-domain registry would ripple a domain param through all three call
# sites for no payoff, since outreach's tool names (lookup_prospect,
# check_suppression, ..., book_survey, apply_discount, ...) don't collide
# with retail's or airline's. Extend-Before-Create: this simply extends the
# existing map. Revisit only if a real cross-domain name collision appears.
#
# Outreach's `place_call`/`send_sms`/`send_email` can each cross EITHER
# contact_without_consent OR out_of_hours_contact (see the cleanroom spec's
# tool-surface table) -- a real two-boundary tool. This dict only supports
# one boundary per tool, so contact_without_consent (the PECR/TPS consent
# check) is recorded as the primary boundary; out_of_hours_contact is not
# reachable via `boundary_for` this slice (it isn't enforced yet -- D6, only
# booking_without_eligibility and unauthorised_incentive are live in slice
# 1). `record_consent` is an audit write, not a boundary (spec), so it has
# no entry.
_TOOL_BOUNDARY: dict[str, str] = {
    "cancel_pending_order": "cancellation",
    "return_delivered_order_items": "refund",
    "modify_pending_order_payment": "payment_change",
    "modify_user_address": "account_change",
    "place_call": "contact_without_consent",
    "send_sms": "contact_without_consent",
    "send_email": "contact_without_consent",
    "book_survey": "booking_without_eligibility",
    "apply_discount": "unauthorised_incentive",
}


@dataclass(frozen=True)
class StandingGrant:
    """A count-only authority grant for one consequence boundary. `max_count` is
    how many times the boundary's action may be taken before it exceeds standing.

    `window` time-bounds that count: `None` (default) is session-scoped (all
    prior actions count); a duration string (e.g. "1h", "30m", "2d") counts only
    actions taken within the window ending at `now`, so authority expires.

    `max_amount` (default `None`) additionally gates by cumulative value moved
    (see the module docstring); `None` means the grant is not amount-gated at
    all -- count is the only ceiling, byte-identical to a v0 grant."""

    boundary: str
    max_count: int
    window: str | None = None
    max_amount: float | None = None


@dataclass(frozen=True)
class TimedAction:
    """A prior action paired with the wall-clock time it was taken (epoch
    seconds). Windowed grants need timing; count-only grants ignore `at`.

    `amount` (default `None`) is the action's resolved value, stamped at
    record time by the caller (the standing_amount oracle resolves it, the
    same moment `at` is stamped) -- `None` for actions that were never
    amount-resolved (e.g. no amount-gated grant was in play)."""

    action: ProposedAction
    at: float
    amount: float | None = None


_WINDOW_UNITS: dict[str, float] = {"s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0}
_WINDOW_RE = re.compile(r"(\d+)([smhd])")


def parse_window(window: str) -> float:
    """Parse a duration string ("45s", "30m", "1h", "2d") into seconds. Raises
    ValueError on anything else — no bare numbers, decimals, signs, or spaces."""
    match = _WINDOW_RE.fullmatch(window)
    if match is None:
        raise ValueError(f"invalid window {window!r}: expected e.g. '45s', '30m', '1h', '2d'")
    value, unit = match.groups()
    return int(value) * _WINDOW_UNITS[unit]


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


def outreach_standing_grants() -> dict[str, StandingGrant]:
    """The outreach (Sunhill) standing policy (bossyk-sandbox slice 1, D5/D7).

    `booking_without_eligibility` is gated by the fast-rule precedence check
    (`scenarios.runner.outreach_fast_rules`), not by standing, so it carries
    no grant here (spec: "gated by the passed-check rule, not standing
    count"). `unauthorised_incentive` is amount-gated: the caller may apply
    discounts up to a fixed cumulative ceiling per session before a further
    discount escalates -- the amount is resolved directly from
    `apply_discount`'s own arguments (no reader lookup needed, unlike
    retail's amount oracle, since the value is already in the tool call)."""
    return {
        "unauthorised_incentive": StandingGrant(
            boundary="unauthorised_incentive", max_count=5, max_amount=150.0
        ),
    }


def boundary_for(tool_name: str) -> str | None:
    """The consequence boundary a tool acts on, or None if it is not a governed
    write (e.g. a read/lookup)."""
    return _TOOL_BOUNDARY.get(tool_name)


def _action_of(item: ProposedAction | TimedAction) -> ProposedAction:
    return item.action if isinstance(item, TimedAction) else item


def _consumes_authority(
    item: ProposedAction | TimedAction,
    boundary: str,
    window: float | None,
    now: float | None,
) -> bool:
    """Whether a prior `item` counts against `boundary`'s standing. Session-scoped
    grants (no window) count every boundary match. A windowed grant counts a match
    only if it falls within `(now - window, now]`. When the window can't be applied
    -- no `now`, or an un-timed item -- the match still counts: missing time must
    never grant *more* authority, so we err toward escalation."""
    if boundary_for(_action_of(item).tool_name) != boundary:
        return False
    if window is None:
        return True
    at = item.at if isinstance(item, TimedAction) else None
    if now is None or at is None:
        return True
    return now - at < window


def _consumed_amount(
    item: ProposedAction | TimedAction,
    boundary: str,
    window: float | None,
    now: float | None,
) -> float:
    """The wall-clock-filtered amount `item` contributes to `boundary`'s
    cumulative spend, mirroring `_consumes_authority`'s window filtering
    exactly. An item that doesn't consume authority at all (wrong boundary, or
    expired under a windowed grant) contributes nothing; one that does but
    carries no stamped amount (a bare `ProposedAction`, or a `TimedAction`
    whose amount was never resolved) also contributes nothing -- an unknown
    historical amount is never invented, it simply can't inflate the budget."""
    if not _consumes_authority(item, boundary, window, now):
        return 0.0
    amount = item.amount if isinstance(item, TimedAction) else None
    return amount if amount is not None else 0.0


def evaluate_authority(
    proposed: ProposedAction,
    history: Sequence[ProposedAction | TimedAction],
    grants: dict[str, StandingGrant],
    *,
    now: float | None = None,
    amount: float | None = None,
) -> AuthorityVerdict:
    """Authority for `proposed`: count-only by default, optionally ALSO
    amount-gated (cumulative budget), optionally time-bounded per grant.

    Counts prior actions in `history` that hit the same boundary; the proposed
    action is the (count+1)-th. If that exceeds the boundary's `max_count` it
    is `over`. If the grant is also amount-gated (`max_amount` is not `None`),
    the proposed action's resolved `amount` (from `standing_amount.resolve_amount`,
    threaded in by the caller) is summed with the in-window consumed amounts
    already stamped on `history`'s `TimedAction`s; exceeding `max_amount` is
    ALSO `over` -- a grant is `over` if EITHER ceiling trips. An unresolvable
    `amount` (`None`) on an amount-gated grant is always `over` (fail-safe:
    it can never prove it's within budget). A tool with no boundary, or a
    boundary with no grant, is `not_governed` (the caller falls back to its
    non-authority heuristic). `history` is whatever the caller deems
    authority-consuming (e.g. the session's allowed actions).

    For a **windowed** grant, pass timed `history` (`TimedAction`) and `now`
    (epoch seconds): only actions within the window ending at `now` count
    towards EITHER ceiling, so standing (count and budget alike) expires.
    Session-scoped grants ignore timing entirely. A grant with
    `max_amount=None` (the default) is not amount-gated at all -- passing
    `amount` has no effect, keeping evaluation byte-identical to a v0,
    count-only grant.
    """
    boundary = boundary_for(proposed.tool_name)
    if boundary is None:
        return AuthorityVerdict("not_governed", f"{proposed.tool_name} is not a governed boundary")

    grant = grants.get(boundary)
    if grant is None:
        return AuthorityVerdict("not_governed", f"no standing grant for {boundary}", boundary)

    window = parse_window(grant.window) if grant.window is not None else None
    nth = sum(1 for item in history if _consumes_authority(item, boundary, window, now)) + 1
    over_count = nth > grant.max_count
    count_reason = f"{boundary}: action #{nth} {{status}} standing max_count {grant.max_count}"

    over_amount = False
    amount_reason = ""
    if grant.max_amount is not None:
        if amount is None:
            # Fail-safe (decision #2): an amount-gated grant with an
            # unresolvable proposed amount can never prove it's within
            # budget, so it is always `over` -- never silently allowed.
            over_amount = True
            amount_reason = (
                f"{boundary}: amount unresolvable for an amount-gated grant "
                f"(max_amount {grant.max_amount}) -- escalating (fail-safe)"
            )
        else:
            consumed = sum(_consumed_amount(item, boundary, window, now) for item in history)
            total = consumed + amount
            over_amount = total > grant.max_amount
            status = "exceeds" if over_amount else "within"
            amount_reason = (
                f"{boundary}: cumulative amount {total} {status} standing "
                f"max_amount {grant.max_amount}"
            )

    if over_count or over_amount:
        parts = [count_reason.format(status="exceeds")] if over_count else []
        if amount_reason:
            parts.append(amount_reason)
        return AuthorityVerdict("over", " and ".join(parts), boundary)

    parts = [count_reason.format(status="within")]
    if amount_reason:
        parts.append(amount_reason)
    return AuthorityVerdict("within", " and ".join(parts), boundary)
