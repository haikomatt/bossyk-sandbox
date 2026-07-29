from __future__ import annotations

import pytest

from bossyk_sandbox.instruments.base import ProposedAction
from bossyk_sandbox.standing import (
    StandingGrant,
    TimedAction,
    boundary_for,
    evaluate_authority,
    parse_window,
)

# §F v0 (decisions locked 2026-07-28): count-only authority -- "how many times a
# session may hit a consequence boundary before it exceeds standing"; history is
# the session's prior actions. Session-scoped by default; a grant may carry a
# `window` to time-bound standing (see the wall-clock section below). Amount
# gating is still a deliberate follow-up.

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


# --- wall-clock windows (job #1) ---------------------------------------------
# A windowed grant time-bounds standing: authority *expires*, so a prior action
# older than the window no longer consumes it. `now` is injected (never read
# from a real clock here) to keep evaluation pure and deterministic; a windowed
# grant counts only history whose timestamp falls within `(now - window, now]`.

# "2 cancels per hour", vs the session-scoped GRANTS above ("2 per session").
HOURLY = {"cancellation": StandingGrant(boundary="cancellation", max_count=2, window="1h")}


def _timed_cancel(at: float, order_id: str = "#W1") -> TimedAction:
    return TimedAction(_cancel(order_id), at)


def test_parse_window_units() -> None:
    assert parse_window("45s") == 45.0
    assert parse_window("30m") == 30 * 60.0
    assert parse_window("2h") == 2 * 3600.0
    assert parse_window("1d") == 86400.0


def test_parse_window_rejects_garbage() -> None:
    for bad in ("", "h", "1", "1x", "1.5h", "-1h", "1 h", "60"):
        with pytest.raises(ValueError):
            parse_window(bad)


def test_window_field_defaults_to_none() -> None:
    # An un-windowed grant is session-scoped: existing construction is unchanged.
    assert StandingGrant(boundary="cancellation", max_count=2).window is None


def test_expired_actions_do_not_consume_authority() -> None:
    now = 10_000.0
    # two cancels, both OLDER than an hour (3600s) -> expired, don't count.
    history = [_timed_cancel(now - 4000), _timed_cancel(now - 3601)]
    verdict = evaluate_authority(_cancel(), history, HOURLY, now=now)
    assert verdict.status == "within"  # proposed is the only in-window cancel


def test_actions_inside_the_window_still_consume_authority() -> None:
    now = 10_000.0
    # two cancels within the last hour -> the proposed (3rd in-window) is over.
    history = [_timed_cancel(now - 100), _timed_cancel(now - 3599)]
    verdict = evaluate_authority(_cancel(), history, HOURLY, now=now)
    assert verdict.status == "over"
    assert verdict.boundary == "cancellation"


def test_window_boundary_is_exclusive_at_the_far_edge() -> None:
    now = 10_000.0
    # exactly one window old -> outside (window is (now - w, now], age 3600 == w).
    history = [_timed_cancel(now - 3600), _timed_cancel(now - 3600)]
    assert evaluate_authority(_cancel(), history, HOURLY, now=now).status == "within"


def test_only_in_window_actions_count_mixed() -> None:
    now = 10_000.0
    # one expired + one fresh -> proposed is the 2nd in-window -> still within (max 2)
    history = [_timed_cancel(now - 5000), _timed_cancel(now - 10)]
    assert evaluate_authority(_cancel(), history, HOURLY, now=now).status == "within"


def test_non_windowed_grant_ignores_timestamps() -> None:
    # Back-compat: a grant with no window counts all history regardless of time
    # and needs no `now`.
    old = [_timed_cancel(0.0), _timed_cancel(1.0)]
    assert evaluate_authority(_cancel(), old, GRANTS).status == "over"


def test_windowed_grant_without_now_degrades_conservatively() -> None:
    # Missing time info must not silently grant MORE authority: with a windowed
    # grant but no `now`, un-timed prior actions still count (err toward escalate).
    history = [_cancel(), _cancel()]
    assert evaluate_authority(_cancel(), history, HOURLY).status == "over"
