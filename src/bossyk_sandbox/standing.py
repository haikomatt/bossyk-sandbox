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
evaluation stays pure and deterministic. Amount ceilings remain a deliberate
follow-up (need an amount/outcome oracle; see the plan). Kept proprietary, not
proposed to auditk-spec (per demonstrator §13); the shape is a v0.2 candidate if
a second implementer appears.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
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
    """A count-only authority grant for one consequence boundary. `max_count` is
    how many times the boundary's action may be taken before it exceeds standing.

    `window` time-bounds that count: `None` (default) is session-scoped (all
    prior actions count); a duration string (e.g. "1h", "30m", "2d") counts only
    actions taken within the window ending at `now`, so authority expires."""

    boundary: str
    max_count: int
    window: str | None = None


@dataclass(frozen=True)
class TimedAction:
    """A prior action paired with the wall-clock time it was taken (epoch
    seconds). Windowed grants need timing; count-only grants ignore `at`."""

    action: ProposedAction
    at: float


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


def evaluate_authority(
    proposed: ProposedAction,
    history: Sequence[ProposedAction | TimedAction],
    grants: dict[str, StandingGrant],
    *,
    now: float | None = None,
) -> AuthorityVerdict:
    """Count-only authority for `proposed`, optionally time-bounded per grant.

    Counts prior actions in `history` that hit the same boundary; the proposed
    action is the (count+1)-th. If that exceeds the boundary's grant it is
    `over`, else `within`. A tool with no boundary, or a boundary with no grant,
    is `not_governed` (the caller falls back to its non-authority heuristic).
    `history` is whatever the caller deems authority-consuming (e.g. the
    session's allowed actions).

    For a **windowed** grant, pass timed `history` (`TimedAction`) and `now`
    (epoch seconds): only actions within the window ending at `now` count, so
    standing expires. Session-scoped grants ignore timing entirely.
    """
    boundary = boundary_for(proposed.tool_name)
    if boundary is None:
        return AuthorityVerdict("not_governed", f"{proposed.tool_name} is not a governed boundary")

    grant = grants.get(boundary)
    if grant is None:
        return AuthorityVerdict("not_governed", f"no standing grant for {boundary}", boundary)

    window = parse_window(grant.window) if grant.window is not None else None
    nth = sum(1 for item in history if _consumes_authority(item, boundary, window, now)) + 1
    if nth > grant.max_count:
        return AuthorityVerdict(
            "over",
            f"{boundary}: action #{nth} exceeds standing max_count {grant.max_count}",
            boundary,
        )
    return AuthorityVerdict(
        "within", f"{boundary}: action #{nth} within standing max_count {grant.max_count}", boundary
    )
