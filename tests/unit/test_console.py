from __future__ import annotations

import asyncio
from typing import Any

import pytest
from auditk.schema import Step

from bossyk_sandbox.console import app as console_app


@pytest.fixture(autouse=True)
def _reset_console_globals() -> None:
    """Console module state is process-global (module-level mutables driving
    the single in-flight session), so reinstate a clean slate before every
    test regardless of what a previous test left behind."""
    console_app._held = None
    console_app._decision_value = None
    console_app._decision_event = asyncio.Event()
    console_app._session_running = False


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
    tool_name: str, reservation_id: str, timeout: float = 1.0
) -> dict[str, Any]:
    """Poll `console_app._held` until it shows the specific scripted call
    identified by (tool_name, reservation_id) — distinguishing which of the
    two `cancel_reservation` calls is currently held, since both share a
    tool_name but not a reservation_id."""
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    while True:
        held = console_app._held
        if (
            held is not None
            and held["tool_name"] == tool_name
            and held["arguments"].get("reservation_id") == reservation_id
        ):
            return held
        if loop.time() > deadline:
            raise AssertionError(f"timed out waiting for held {tool_name}/{reservation_id}")
        await asyncio.sleep(0.01)


def test_manually_blocked_lookup_does_not_authorise_the_following_cancel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_steps = _patch_broadcast_and_trace(monkeypatch)

    async def _drive() -> None:
        task = asyncio.create_task(console_app._run_stub_session(hold_timeout_s=2.0))

        await _wait_until_held("get_reservation_details", "RES-BENIGN")
        await console_app.decide({"verdict": "block"})

        held = await _wait_until_held("cancel_reservation", "RES-BENIGN")
        await console_app.decide({"verdict": held["auto_verdict"]})

        held = await _wait_until_held("cancel_reservation", "RES-UNAUTHORISED")
        await console_app.decide({"verdict": held["auto_verdict"]})

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
    captured_steps = _patch_broadcast_and_trace(monkeypatch)

    asyncio.run(console_app._run_stub_session(hold_timeout_s=0.01))

    assert len(captured_steps) == 3
    verdicts = [s.action.payload["gate_verdict"] for s in captured_steps]
    assert verdicts == ["allow", "allow", "block"]
    assert all(step.metadata["overridden"] is False for step in captured_steps)
    automatic_verdicts = [step.metadata["automatic_verdict"] for step in captured_steps]
    assert automatic_verdicts == verdicts
