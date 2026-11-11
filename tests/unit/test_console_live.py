from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest
from auditk.schema import Step
from langchain_core.messages import AIMessage

from bossyk_sandbox.console import app as console_app
from bossyk_sandbox.runtime.langgraph_agent import build_retail_agent_session

# Exercises the console's LIVE broadcast wrapper offline: a fake LLM + fake tau2
# env drive the REAL graph, so no API key and no cost. The billable path is only
# the /session/live endpoint invoked with a real agent key, which is never run
# here (the endpoint's no-key branch is tested via a raising build double).


@pytest.fixture(autouse=True)
def _reset_console_globals() -> None:
    console_app._sessions.clear()


class _FakeToolkit:
    def use_tool(self, name: str, **args: Any) -> str:
        return f"ok:{name}"

    def get_tools(self) -> dict[str, Any]:
        return {}


class _FakeEnv:
    policy = "You are a retail support agent."
    tools = _FakeToolkit()


class _FakeLLM:
    def __init__(self, responses: list[AIMessage]) -> None:
        self._responses = responses
        self._i = 0

    def invoke(self, _messages: Any) -> AIMessage:
        response = self._responses[min(self._i, len(self._responses) - 1)]
        self._i += 1
        return response


def _scripted_agent() -> Any:
    calls = [
        {"name": "get_order_details", "args": {"order_id": "#W1"}, "id": "c1"},
        {"name": "cancel_pending_order", "args": {"order_id": "#W2"}, "id": "c2"},
        {"name": "return_delivered_order_items", "args": {"order_id": "#W3"}, "id": "c3"},
        {"name": "get_user_details", "args": {"user_id": "U1"}, "id": "c4"},
        {"name": "modify_pending_order_payment", "args": {"order_id": "#W4"}, "id": "c5"},
    ]
    llm = _FakeLLM([AIMessage(content="", tool_calls=calls), AIMessage(content="done")])
    return build_retail_agent_session(llm=llm, environment=_FakeEnv())


def _patch_broadcast(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []

    async def _fake_broadcast(event: dict[str, Any]) -> None:
        events.append(event)

    def _fake_build_trace(**_kwargs: Any) -> None:
        return None

    monkeypatch.setattr(console_app, "_broadcast", _fake_broadcast)
    monkeypatch.setattr(console_app, "build_trace", _fake_build_trace)
    return events


def test_live_session_broadcasts_derived_modes_and_hitl_queue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = _patch_broadcast(monkeypatch)
    agent_session = _scripted_agent()

    async def _drive() -> None:
        session = console_app.ConsoleSession(session_id="live-test")
        console_app._sessions["live-test"] = session
        await console_app._run_live_session(session, agent_session, pacing_s=0.0)

    asyncio.run(_drive())

    held = [e for e in events if e["type"] == "held"]
    steps = [e for e in events if e["type"] == "step"]
    complete = next(e for e in events if e["type"] == "session_complete")

    assert len(held) == 5
    assert [s["mode"] for s in steps] == ["allow", "redirect", "escalate", "step-up", "escalate"]
    assert [s["verdict"] for s in steps] == ["allow", "block", "block", "allow", "block"]
    assert all(e.get("live") is True for e in held)

    # Compliance controls ride the REAL attested steps the graph produced.
    assert any(s["controls"] for s in steps)

    escalated = [s for s in steps if s["mode"] == "escalate"]
    assert len(escalated) == 2
    assert all(s.get("hitl") and s["hitl"]["severity"] for s in escalated)

    assert [item["severity"] for item in complete["hitl_queue"]] == ["critical", "high"]
    assert complete["live"] is True


def test_live_session_frees_the_slot_when_done(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_broadcast(monkeypatch)
    agent_session = _scripted_agent()

    async def _drive() -> bool:
        session = console_app.ConsoleSession(session_id="live-test")
        console_app._sessions["live-test"] = session
        await console_app._run_live_session(session, agent_session, pacing_s=0.0)
        return "live-test" in console_app._sessions

    assert asyncio.run(_drive()) is False


def test_start_live_session_is_disabled_without_the_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The hard gate: without RUN_LIVE_CONSOLE=1 the endpoint must refuse BEFORE
    # building the session or resolving any key, so a key in the environment is
    # never sufficient and a stray click/POST can never bill.
    monkeypatch.delenv("RUN_LIVE_CONSOLE", raising=False)

    def _must_not_build(**_kwargs: Any) -> Step:
        raise AssertionError("live session must not be built when live is disabled")

    monkeypatch.setattr(console_app, "build_weakened_retail_agent_session", _must_not_build)

    async def _drive() -> dict[str, str]:
        return await console_app.start_live_session(pacing_s=0.0)

    result = asyncio.run(_drive())
    assert result == {"status": "live_disabled"}
    assert not console_app._sessions


def test_start_live_session_reports_no_api_key_when_enabled_but_keyless(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_broadcast(monkeypatch)
    monkeypatch.setenv("RUN_LIVE_CONSOLE", "1")

    def _raise(**_kwargs: Any) -> Step:
        raise RuntimeError("No agent API key")

    monkeypatch.setattr(console_app, "build_weakened_retail_agent_session", _raise)

    async def _drive() -> dict[str, str]:
        return await console_app.start_live_session(pacing_s=0.0)

    result = asyncio.run(_drive())
    assert result == {"status": "no_api_key"}
    assert not console_app._sessions  # nothing scheduled


# --- Phase A: escalation is enforced in the async live loop too ---------------
# Same contract as `test_live_session.py`'s Phase A block, for the console's
# own loop (`console_app._run_live_session`): an `escalate` with no approver
# resolves to BLOCK (decision D1), so the action never executes, its signed
# step and its step event say `block` with mode `escalate`, and it does not
# consume standing authority. `defer` still executes.


class _RecordingToolkit:
    """Records every executed call; optionally carries `.db.orders` (as the
    real `RetailTools` does) so `order_reader_from_toolkit` can price refunds."""

    def __init__(self, orders: dict[str, Any] | None = None) -> None:
        self.db = SimpleNamespace(orders=orders or {})
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def use_tool(self, name: str, **args: Any) -> str:
        self.calls.append((name, args))
        return f"ok:{name}"

    def get_tools(self) -> dict[str, Any]:
        return {}


class _RecordingEnv:
    policy = "You are a retail support agent."

    def __init__(self, orders: dict[str, Any] | None = None) -> None:
        self.tools = _RecordingToolkit(orders)


def _priced_order(item_id: str, price: float) -> Any:
    return SimpleNamespace(
        items=[SimpleNamespace(item_id=item_id, price=price)], payment_history=[]
    )


def _looked_up_calls(tool: str, order_ids: list[str]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for n, order_id in enumerate(order_ids, start=1):
        args: dict[str, Any] = {"order_id": order_id}
        if tool == "return_delivered_order_items":
            args |= {"item_ids": [f"item-{n}"], "payment_method_id": "gift_card_1"}
        calls.append({"name": "get_order_details", "args": {"order_id": order_id}, "id": f"l{n}"})
        calls.append({"name": tool, "args": args, "id": f"w{n}"})
    return calls


def _drive_live(
    monkeypatch: pytest.MonkeyPatch,
    calls: list[dict[str, Any]],
    env: _RecordingEnv,
    *,
    reader: Any = None,
) -> tuple[list[dict[str, Any]], list[Step]]:
    events = _patch_broadcast(monkeypatch)
    llm = _FakeLLM([AIMessage(content="", tool_calls=calls), AIMessage(content="done")])
    agent_session = build_retail_agent_session(llm=llm, environment=env)

    async def _drive() -> None:
        session = console_app.ConsoleSession(session_id="live-test")
        console_app._sessions["live-test"] = session
        await console_app._run_live_session(session, agent_session, reader=reader, pacing_s=0.0)

    asyncio.run(_drive())
    return events, list(agent_session.steps)


def _executed(env: _RecordingEnv, tool: str) -> list[str]:
    return [str(args["order_id"]) for name, args in env.tools.calls if name == tool]


def test_live_console_never_executes_an_escalated_refund(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # retail_standing_grants(): refund max_count=1 -> the 2nd refund is over
    # standing and irreversible -> escalate.
    env = _RecordingEnv()
    events, _steps = _drive_live(
        monkeypatch, _looked_up_calls("return_delivered_order_items", ["#R1", "#R2"]), env
    )

    steps = [e for e in events if e["type"] == "step"]
    assert [s["mode"] for s in steps] == ["allow", "allow", "allow", "escalate"]
    assert _executed(env, "return_delivered_order_items") == ["#R1"]


def test_live_console_signs_an_escalated_refund_as_a_standing_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _RecordingEnv()
    events, steps = _drive_live(
        monkeypatch, _looked_up_calls("return_delivered_order_items", ["#R1", "#R2"]), env
    )

    escalated = steps[-1]
    assert escalated.action.payload["gate_verdict"] == "block"
    assert escalated.metadata["automatic_verdict"] == "allow"
    assert escalated.metadata["mode"] == "escalate"
    assert escalated.metadata["resolution_source"] == "standing"
    assert escalated.metadata["overridden"] is False

    # The held event still shows what the gate itself said; the step event
    # (what the control room renders as the outcome) shows the final verdict.
    held = [e for e in events if e["type"] == "held"]
    step_events = [e for e in events if e["type"] == "step"]
    assert held[-1]["auto_verdict"] == "allow"
    assert step_events[-1]["verdict"] == "block"
    assert step_events[-1]["mode"] == "escalate"
    assert step_events[-1]["overridden"] is False


def test_live_console_escalated_refund_does_not_consume_standing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Amount-gated grant: 60 ok, +60 = 120 > 100 escalates (blocked), then 30
    # must see 60 + 30 = 90, not 150.
    from bossyk_sandbox.console.live import order_reader_from_toolkit
    from bossyk_sandbox.standing import StandingGrant

    monkeypatch.setattr(
        console_app,
        "retail_standing_grants",
        lambda: {"refund": StandingGrant(boundary="refund", max_count=10, max_amount=100.0)},
    )
    orders = {
        "#R1": _priced_order("item-1", 60.0),
        "#R2": _priced_order("item-2", 60.0),
        "#R3": _priced_order("item-3", 30.0),
    }
    env = _RecordingEnv(orders)
    events, _steps = _drive_live(
        monkeypatch,
        _looked_up_calls("return_delivered_order_items", ["#R1", "#R2", "#R3"]),
        env,
        reader=order_reader_from_toolkit(env.tools),
    )

    steps = [e for e in events if e["type"] == "step"]
    assert [s["mode"] for s in steps] == ["allow", "allow", "allow", "escalate", "allow", "allow"]
    assert _executed(env, "return_delivered_order_items") == ["#R1", "#R3"]


def test_live_console_still_executes_a_deferred_cancel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # retail_standing_grants(): cancellation max_count=2 -> the 3rd cancel is
    # over standing but reversible -> defer (accept + async-approve).
    env = _RecordingEnv()
    events, steps = _drive_live(
        monkeypatch, _looked_up_calls("cancel_pending_order", ["#A", "#B", "#C"]), env
    )

    step_events = [e for e in events if e["type"] == "step"]
    assert step_events[-1]["mode"] == "defer"
    assert step_events[-1]["verdict"] == "allow"
    assert _executed(env, "cancel_pending_order") == ["#A", "#B", "#C"]
    assert steps[-1].action.payload["gate_verdict"] == "allow"
