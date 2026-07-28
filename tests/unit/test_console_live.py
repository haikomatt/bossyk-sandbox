from __future__ import annotations

import asyncio
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


def test_start_live_session_reports_no_api_key_without_running_billably(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_broadcast(monkeypatch)

    def _raise(**_kwargs: Any) -> Step:
        raise RuntimeError("No agent API key")

    monkeypatch.setattr(console_app, "build_weakened_retail_agent_session", _raise)

    async def _drive() -> dict[str, str]:
        return await console_app.start_live_session(pacing_s=0.0)

    result = asyncio.run(_drive())
    assert result == {"status": "no_api_key"}
    assert not console_app._sessions  # nothing scheduled
