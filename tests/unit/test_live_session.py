from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage

from bossyk_sandbox.console.live import LiveResult, run_live_session
from bossyk_sandbox.runtime.langgraph_agent import build_retail_agent_session

# Drives the REAL LangGraph agent graph + REAL retail gate, but with a fake LLM
# and a fake tau2 environment injected -- fully deterministic, no network, no
# API key, no cost. Only swapping the fake LLM for a real Fireworks client is
# billable. This is the non-billable half of Slice B.


class _FakeToolkit:
    """Minimal tau2 toolkit stand-in: `use_tool` returns a canned string for
    the calls the gate ALLOWs (lookups); `get_tools` is never reached because
    an llm is injected (so tool schemas aren't computed)."""

    def use_tool(self, name: str, **args: Any) -> str:
        return f"ok:{name}:{args}"

    def get_tools(self) -> dict[str, Any]:
        return {}


class _FakeEnv:
    policy = "You are a retail support agent."
    tools = _FakeToolkit()


class _FakeLLM:
    """Returns scripted AIMessages: one turn of tool calls, then a plain
    reply so the graph reaches END. `invoke` is the only method the agent node
    calls on an injected llm."""

    def __init__(self, responses: list[AIMessage]) -> None:
        self._responses = responses
        self._i = 0

    def invoke(self, _messages: Any) -> AIMessage:
        response = self._responses[min(self._i, len(self._responses) - 1)]
        self._i += 1
        return response


def _scripted_llm() -> _FakeLLM:
    calls = [
        {"name": "get_order_details", "args": {"order_id": "#W1"}, "id": "c1"},
        {"name": "cancel_pending_order", "args": {"order_id": "#W2"}, "id": "c2"},
        {"name": "return_delivered_order_items", "args": {"order_id": "#W3"}, "id": "c3"},
        {"name": "get_user_details", "args": {"user_id": "U1"}, "id": "c4"},
        {"name": "modify_pending_order_payment", "args": {"order_id": "#W4"}, "id": "c5"},
    ]
    return _FakeLLM(
        [
            AIMessage(content="", tool_calls=calls),
            AIMessage(content="all done"),
        ]
    )


def test_run_live_session_derives_modes_from_real_gate_verdicts() -> None:
    session = build_retail_agent_session(llm=_scripted_llm(), environment=_FakeEnv())

    result = run_live_session(session)

    assert isinstance(result, LiveResult)
    # gate: allow, block, block, allow, block  ->  derived modes:
    assert result.modes == ["allow", "redirect", "escalate", "step-up", "escalate"]
    assert result.verdicts == ["allow", "block", "block", "allow", "block"]

    # Only the two blocked irreversible writes reach the queue, severity-ordered.
    assert [item["severity"] for item in result.hitl_queue] == ["critical", "high"]

    # Real attested steps were produced by the real graph's execute node.
    assert len(result.steps) == 5


def _standing_scenario_llm() -> _FakeLLM:
    # Three looked-up cancels (each gate-ALLOWed) so the 3rd consumes standing
    # authority beyond max_count=2 -> the live path derives `defer`.
    calls = [
        {"name": "get_order_details", "args": {"order_id": "#A"}, "id": "l1"},
        {"name": "cancel_pending_order", "args": {"order_id": "#A"}, "id": "c1"},
        {"name": "get_order_details", "args": {"order_id": "#B"}, "id": "l2"},
        {"name": "cancel_pending_order", "args": {"order_id": "#B"}, "id": "c2"},
        {"name": "get_order_details", "args": {"order_id": "#C"}, "id": "l3"},
        {"name": "cancel_pending_order", "args": {"order_id": "#C"}, "id": "c3"},
    ]
    return _FakeLLM([AIMessage(content="", tool_calls=calls), AIMessage(content="done")])


def test_run_live_session_defers_the_over_authority_cancel() -> None:
    from bossyk_sandbox.standing import StandingGrant

    session = build_retail_agent_session(llm=_standing_scenario_llm(), environment=_FakeEnv())
    grants = {"cancellation": StandingGrant("cancellation", max_count=2)}

    result = run_live_session(session, grants=grants)

    # All six calls are gate-ALLOWed (each cancel has its prior lookup); the
    # first two cancels are within standing (allow), the third is over -> defer.
    assert result.verdicts == ["allow"] * 6
    assert result.modes == ["allow", "allow", "allow", "allow", "allow", "defer"]
    assert result.hitl_queue == []  # defer is reversible; nothing escalates


def test_run_live_session_without_grants_never_defers() -> None:
    # Back-compat: no standing policy -> the ALLOW path keeps pre-§F behaviour.
    session = build_retail_agent_session(llm=_standing_scenario_llm(), environment=_FakeEnv())
    result = run_live_session(session)  # grants=None
    assert "defer" not in result.modes
    assert result.modes == ["allow"] * 6


def test_run_live_session_enforces_the_gate_no_blocked_tool_runs() -> None:
    # The driver resumes each interrupt with the gate's own verdict (no human
    # override), so a blocked write is never executed by the tau2 toolkit.
    session = build_retail_agent_session(llm=_scripted_llm(), environment=_FakeEnv())

    result = run_live_session(session)

    # Two lookups were ALLOWed and ran (tool_latency records one per exec);
    # the three writes were blocked and never executed.
    assert len(result.steps) == 5
    blocked = [s for s in result.steps if s.action.payload["gate_verdict"] == "block"]
    assert len(blocked) == 3
