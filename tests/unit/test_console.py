from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

import pytest
from auditk.schema import Step

from bossyk_sandbox.console import app as console_app


@pytest.fixture(autouse=True)
def _reset_console_globals() -> None:
    """Console module state is process-global (the `_sessions` registry
    driving the single in-flight session), so reinstate a clean slate before
    every test regardless of what a previous test left behind."""
    console_app._sessions.clear()


def _patch_broadcast_and_trace(monkeypatch: pytest.MonkeyPatch) -> list[Step]:
    """Silence websocket broadcast (no client is connected in-process) and
    capture the steps `_run_stub_session` hands to `build_trace` so tests
    can inspect their gate/metadata contents directly."""
    captured_steps: list[Step] = []

    async def _fake_broadcast(_event: dict[str, Any]) -> None:
        return None

    def _fake_build_trace(*, trace_id: str, agent_config_ref: str, steps: list[Step]) -> None:
        captured_steps.extend(steps)

    monkeypatch.setattr(console_app, "_broadcast", _fake_broadcast)
    monkeypatch.setattr(console_app, "build_trace", _fake_build_trace)
    return captured_steps


async def _wait_until_held(
    session_id: str, tool_name: str, reservation_id: str, timeout: float = 1.0
) -> console_app.HeldAction:
    """Poll `_sessions[session_id].held` until it shows the specific
    scripted call identified by (tool_name, reservation_id) —
    distinguishing which of the two `cancel_reservation` calls is currently
    held, since both share a tool_name but not a reservation_id."""
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    while True:
        session = console_app._sessions.get(session_id)
        held = session.held if session is not None else None
        if (
            held is not None
            and held.payload["tool_name"] == tool_name
            and held.payload["arguments"].get("reservation_id") == reservation_id
        ):
            return held
        if loop.time() > deadline:
            raise AssertionError(f"timed out waiting for held {tool_name}/{reservation_id}")
        await asyncio.sleep(0.01)


def test_manually_blocked_lookup_does_not_authorise_the_following_cancel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Finding 6: migrated to the session-scoped start_session/decide API --
    # the old test drove `_run_stub_session` directly and decided against
    # the single implicit `_held` slot, both since deleted.
    captured_steps = _patch_broadcast_and_trace(monkeypatch)

    async def _drive() -> None:
        start_result = await console_app.start_session(hold_timeout_s=2.0)
        session_id = start_result["session_id"]
        task = console_app._sessions[session_id].task
        assert task is not None

        held = await _wait_until_held(session_id, "get_reservation_details", "RES-BENIGN")
        await console_app.decide(
            {"session_id": session_id, "action_id": held.action_id, "verdict": "block"}
        )

        held = await _wait_until_held(session_id, "cancel_reservation", "RES-BENIGN")
        await console_app.decide(
            {
                "session_id": session_id,
                "action_id": held.action_id,
                "verdict": held.payload["auto_verdict"],
            }
        )

        held = await _wait_until_held(session_id, "cancel_reservation", "RES-UNAUTHORISED")
        await console_app.decide(
            {
                "session_id": session_id,
                "action_id": held.action_id,
                "verdict": held.payload["auto_verdict"],
            }
        )

        await task

    asyncio.run(_drive())

    lookup_step = next(
        s for s in captured_steps if s.action.payload["tool_name"] == "get_reservation_details"
    )
    cancel_benign_step = next(
        s
        for s in captured_steps
        if s.action.payload["tool_name"] == "cancel_reservation"
        and s.action.payload["arguments"]["reservation_id"] == "RES-BENIGN"
    )

    # The manually blocked lookup must never have entered gate history, so
    # the benign cancel's automatic score can't find a prior lookup either.
    assert cancel_benign_step.action.payload["gate_verdict"] == "block"
    assert lookup_step.metadata["overridden"] is True
    assert lookup_step.metadata["automatic_verdict"] == "allow"


def test_untouched_session_attests_automatic_verdicts(monkeypatch: pytest.MonkeyPatch) -> None:
    # Finding 6: migrated to start_session -- the old test drove
    # `_run_stub_session` directly with no session argument.
    captured_steps = _patch_broadcast_and_trace(monkeypatch)

    async def _drive() -> None:
        start_result = await console_app.start_session(hold_timeout_s=0.01)
        task = console_app._sessions[start_result["session_id"]].task
        assert task is not None
        await task

    asyncio.run(_drive())

    assert len(captured_steps) == 3
    verdicts = [s.action.payload["gate_verdict"] for s in captured_steps]
    assert verdicts == ["allow", "allow", "block"]
    assert all(step.metadata["overridden"] is False for step in captured_steps)
    automatic_verdicts = [step.metadata["automatic_verdict"] for step in captured_steps]
    assert automatic_verdicts == verdicts


# --- Finding 6: console sessions must be scoped by session_id/action_id,
# not process-global module state --------------------------------------------


def _patch_broadcast_events_and_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[dict[str, Any]], list[Step]]:
    """Like `_patch_broadcast_and_trace`, but also captures every broadcast
    *event* (not just the steps handed to `build_trace`) so new tests can
    inspect the "held" event's identity fields."""
    captured_events: list[dict[str, Any]] = []
    captured_steps: list[Step] = []

    async def _fake_broadcast(event: dict[str, Any]) -> None:
        captured_events.append(event)

    def _fake_build_trace(*, trace_id: str, agent_config_ref: str, steps: list[Step]) -> None:
        captured_steps.extend(steps)

    monkeypatch.setattr(console_app, "_broadcast", _fake_broadcast)
    monkeypatch.setattr(console_app, "build_trace", _fake_build_trace)
    return captured_events, captured_steps


async def _wait_for_held_event(
    events: list[dict[str, Any]],
    matches: Callable[[dict[str, Any]], bool],
    timeout: float = 2.0,
) -> dict[str, Any]:
    """Poll a list of captured broadcast events until one of type "held"
    satisfies `matches`."""
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    while True:
        for event in events:
            if event.get("type") == "held" and matches(event):
                return event
        if loop.time() > deadline:
            raise AssertionError("timed out waiting for a matching held event")
        await asyncio.sleep(0.01)


async def _wait_until_any_held(timeout: float = 2.0) -> dict[str, Any]:
    """Poll `console_app._sessions` (Finding 6: the session-scoped registry
    that replaced the single process-global `_held` slot) until any session
    has a held action, regardless of which scripted call it is. Only one
    session is ever active at a time, so "any" is unambiguous."""
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    while True:
        held = next((s.held for s in console_app._sessions.values() if s.held is not None), None)
        if held is not None:
            return held.payload
        if loop.time() > deadline:
            raise AssertionError("timed out waiting for any held action")
        await asyncio.sleep(0.01)


def test_concurrent_starts_collapse_to_a_single_session(monkeypatch: pytest.MonkeyPatch) -> None:
    """Finding 6: `start_session` only flips `_session_running` inside the
    task it schedules, so two `POST /session/run` calls racing before either
    task has run both see the flag as False and both report "started". The
    session must be registered synchronously, before the task ever runs, so
    a concurrent second call sees it immediately."""
    _patch_broadcast_and_trace(monkeypatch)

    async def _drive() -> tuple[dict[str, str], dict[str, str]]:
        first, second = await asyncio.gather(
            console_app.start_session(hold_timeout_s=0.05),
            console_app.start_session(hold_timeout_s=0.05),
        )
        return first, second

    first, second = asyncio.run(_drive())

    statuses = sorted([first["status"], second["status"]])
    assert statuses == ["already_running", "started"]

    started = first if first["status"] == "started" else second
    assert "session_id" in started


def test_held_broadcast_carries_session_and_action_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_events, _steps = _patch_broadcast_events_and_trace(monkeypatch)

    async def _drive() -> tuple[dict[str, str], dict[str, Any]]:
        start_result = await console_app.start_session(hold_timeout_s=0.05)
        held_event = await _wait_for_held_event(
            captured_events,
            lambda e: (
                e.get("tool_name") == "get_reservation_details"
                and e.get("arguments", {}).get("reservation_id") == "RES-BENIGN"
            ),
        )
        return start_result, held_event

    start_result, held_event = asyncio.run(_drive())

    assert "session_id" in start_result, "start_session must return a session_id"
    session_id = start_result["session_id"]
    assert held_event.get("session_id") == session_id
    assert held_event.get("action_id") == f"{session_id}-action-0"


def test_decide_demands_matching_session_and_action_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_events, captured_steps = _patch_broadcast_events_and_trace(monkeypatch)

    async def _drive() -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
        start_result = await console_app.start_session(hold_timeout_s=0.05)
        assert "session_id" in start_result, "start_session must return a session_id"
        session_id = start_result["session_id"]

        held_event = await _wait_for_held_event(
            captured_events,
            lambda e: (
                e.get("tool_name") == "get_reservation_details"
                and e.get("arguments", {}).get("reservation_id") == "RES-BENIGN"
            ),
        )
        assert "action_id" in held_event, "held broadcast must carry an action_id"
        action_id = held_event["action_id"]

        wrong_session = await console_app.decide(
            {"session_id": f"not-{session_id}", "action_id": action_id, "verdict": "block"}
        )
        stale = await console_app.decide(
            {"session_id": session_id, "action_id": f"{session_id}-action-99", "verdict": "block"}
        )
        matching = await console_app.decide(
            {"session_id": session_id, "action_id": action_id, "verdict": "block"}
        )
        # Let the rest of the scripted session run to completion so the
        # override is attested in the captured steps.
        await asyncio.sleep(0.5)
        return wrong_session, stale, matching

    wrong_session, stale, matching = asyncio.run(_drive())

    assert wrong_session == {"status": "unknown_session"}
    assert stale == {"status": "stale_action"}
    assert matching == {"status": "accepted"}

    lookup_step = next(
        s for s in captured_steps if s.action.payload["tool_name"] == "get_reservation_details"
    )
    assert lookup_step.metadata["overridden"] is True


def test_finished_session_frees_the_console_for_a_new_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_broadcast_and_trace(monkeypatch)

    async def _drive() -> tuple[str, dict[str, str]]:
        first = await console_app.start_session(hold_timeout_s=0.05)
        assert "session_id" in first, "start_session must return a session_id"
        session_id = first["session_id"]

        # Drive the three scripted steps to their auto verdict so the
        # session actually finishes instead of idling on its hold timeout.
        for _ in range(3):
            held = await _wait_until_any_held()
            await console_app.decide(
                {
                    "session_id": session_id,
                    "action_id": "irrelevant-for-this-test",
                    "verdict": held["auto_verdict"],
                }
            )
        await asyncio.sleep(0.2)  # let `_run_stub_session`'s finally block run

        second = await console_app.start_session(hold_timeout_s=0.05)
        return session_id, second

    session_id, second = asyncio.run(_drive())

    assert second["status"] == "started"
    assert second.get("session_id") != session_id
