from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command

from bossyk_sandbox.instruments.base import ProposedAction
from bossyk_sandbox.instruments.hardcoded_rule import RequireLookupBeforeCancel
from bossyk_sandbox.runtime.langgraph_agent import (
    AirlineAgentSession,
    build_airline_agent_session,
    build_retail_agent_session,
    retail_tool_schemas,
)


@dataclass
class _ScriptedLLM:
    """Fake tool-bound chat model: `invoke` pops the next scripted
    `AIMessage` off a queue regardless of what messages it's called with.
    Stands in for a real ChatOpenAI client from the graph's perspective,
    with no network call and no API key."""

    responses: list[AIMessage]
    calls: int = 0

    def invoke(self, _messages: Any) -> AIMessage:
        response = self.responses[self.calls]
        self.calls += 1
        return response


@dataclass
class _CountingToolkit:
    """Fake tau2 toolkit: counts invocations per (tool_name, frozen args)
    and can be configured to raise for specific tool names, to prove a
    failed call never fabricates a successful history entry."""

    raising_tools: frozenset[str] = frozenset()
    counts: dict[tuple[str, tuple[tuple[str, Any], ...]], int] = field(default_factory=dict)

    def get_tools(self) -> dict[str, Any]:
        return {}

    def use_tool(self, name: str, **kwargs: Any) -> str:
        key = (name, tuple(sorted(kwargs.items())))
        self.counts[key] = self.counts.get(key, 0) + 1
        if name in self.raising_tools:
            raise RuntimeError("boom")
        return "ok"

    def call_count(self, name: str, **kwargs: Any) -> int:
        key = (name, tuple(sorted(kwargs.items())))
        return self.counts.get(key, 0)


@dataclass
class _FakeEnvironment:
    """Fake tau2 environment: exposes `.policy` and `.tools`, enough for
    `build_airline_agent_session(environment=...)` to skip the real tau2
    `get_environment()` call entirely."""

    tools: _CountingToolkit
    policy: str = "test policy"


def _tool_call(name: str, args: dict[str, Any], call_id: str) -> dict[str, Any]:
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


def _run_to_completion(
    session: AirlineAgentSession,
    *,
    thread_id: str,
    resume_values: dict[str, str] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Invoke the graph to completion, resuming every interrupt encountered.

    Each interrupt is resumed with `resume_values[tool_call_id]` if present,
    else the interrupt payload's own `auto_verdict` (accepting the automatic
    decision) — mirroring `scripts/live_demo.py`'s driver contract. Returns
    the final graph result plus every interrupt payload observed, in order.
    """
    config = {"configurable": {"thread_id": thread_id}}
    resume_values = resume_values or {}
    payloads: list[dict[str, Any]] = []

    result: dict[str, Any] = session.graph.invoke(  # type: ignore[call-overload]
        {"messages": [HumanMessage(content="please help with my reservation")]}, config=config
    )
    while "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        payloads.append(payload)
        resume = resume_values.get(payload["tool_call_id"], payload["auto_verdict"])
        result = session.graph.invoke(  # type: ignore[call-overload]
            Command(resume=resume), config=config
        )
    return result, payloads


def test_two_tool_calls_in_one_ai_message_each_execute_exactly_once() -> None:
    toolkit = _CountingToolkit()
    llm = _ScriptedLLM(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    _tool_call("get_reservation_details", {"reservation_id": "R1"}, "call-1"),
                    _tool_call("cancel_reservation", {"reservation_id": "R1"}, "call-2"),
                ],
            ),
            AIMessage(content="done"),
        ]
    )
    session = build_airline_agent_session(
        trace_id="t-two-calls", llm=llm, environment=_FakeEnvironment(tools=toolkit)
    )

    _result, payloads = _run_to_completion(session, thread_id="two-calls")

    assert len(payloads) == 2
    assert toolkit.call_count("get_reservation_details", reservation_id="R1") == 1
    assert toolkit.call_count("cancel_reservation", reservation_id="R1") == 1
    assert len(session.steps) == 2
    # The cancel's auto verdict saw the already-executed lookup.
    assert payloads[1]["auto_verdict"] == "allow"
    assert session.gate is not None
    assert session.gate.history == [
        ProposedAction("get_reservation_details", {"reservation_id": "R1"}),
        ProposedAction("cancel_reservation", {"reservation_id": "R1"}),
    ]
    assert all(step.metadata["overridden"] is False for step in session.steps)


def test_manual_allow_over_automatic_block_executes_and_attests_honestly() -> None:
    toolkit = _CountingToolkit()
    llm = _ScriptedLLM(
        responses=[
            AIMessage(
                content="",
                tool_calls=[_tool_call("cancel_reservation", {"reservation_id": "R9"}, "call-1")],
            ),
            AIMessage(content="done"),
        ]
    )
    session = build_airline_agent_session(
        trace_id="t-manual-allow", llm=llm, environment=_FakeEnvironment(tools=toolkit)
    )

    _result, payloads = _run_to_completion(
        session, thread_id="manual-allow", resume_values={"call-1": "allow"}
    )

    assert payloads[0]["auto_verdict"] == "block"
    assert toolkit.call_count("cancel_reservation", reservation_id="R9") == 1
    assert len(session.steps) == 1
    step = session.steps[0]
    assert step.action.payload["gate_verdict"] == "allow"
    assert step.metadata["automatic_verdict"] == "block"
    assert step.metadata["overridden"] is True
    assert session.gate is not None
    assert session.gate.history == [ProposedAction("cancel_reservation", {"reservation_id": "R9"})]


def test_manual_block_over_automatic_allow_does_not_execute_and_attests_honestly() -> None:
    toolkit = _CountingToolkit()
    llm = _ScriptedLLM(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    _tool_call("get_reservation_details", {"reservation_id": "R1"}, "call-1")
                ],
            ),
            AIMessage(content="done"),
        ]
    )
    session = build_airline_agent_session(
        trace_id="t-manual-block", llm=llm, environment=_FakeEnvironment(tools=toolkit)
    )

    result, payloads = _run_to_completion(
        session, thread_id="manual-block", resume_values={"call-1": "block"}
    )

    assert payloads[0]["auto_verdict"] == "allow"
    assert toolkit.call_count("get_reservation_details", reservation_id="R1") == 0
    assert session.gate is not None
    assert session.gate.history == []
    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert any(
        isinstance(m.content, str) and m.content.startswith("BLOCKED by bossyk-sandbox GATE:")
        for m in tool_messages
    )
    step = session.steps[0]
    assert step.action.payload["gate_verdict"] == "block"
    assert step.metadata["automatic_verdict"] == "allow"
    assert step.metadata["overridden"] is True


def test_a_tool_that_raises_never_enters_history() -> None:
    toolkit = _CountingToolkit(raising_tools=frozenset({"get_reservation_details"}))
    llm = _ScriptedLLM(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    _tool_call("get_reservation_details", {"reservation_id": "R1"}, "call-1"),
                    _tool_call("cancel_reservation", {"reservation_id": "R1"}, "call-2"),
                ],
            ),
            AIMessage(content="done"),
        ]
    )
    session = build_airline_agent_session(
        trace_id="t-raises", llm=llm, environment=_FakeEnvironment(tools=toolkit)
    )

    result, payloads = _run_to_completion(session, thread_id="raises")

    # The failed lookup did not authorise the cancel.
    assert payloads[1]["auto_verdict"] == "block"
    assert session.gate is not None
    assert session.gate.history == []
    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert any(isinstance(m.content, str) and m.content.startswith("error:") for m in tool_messages)


def test_no_interrupts_when_the_llm_proposes_no_tools() -> None:
    toolkit = _CountingToolkit()
    llm = _ScriptedLLM(responses=[AIMessage(content="done")])
    session = build_airline_agent_session(
        trace_id="t-no-tools", llm=llm, environment=_FakeEnvironment(tools=toolkit)
    )

    result, payloads = _run_to_completion(session, thread_id="no-tools")

    assert payloads == []
    assert "__interrupt__" not in result
    assert session.steps == []
    assert session.gate is not None
    assert session.gate.history == []


# --- domain-parameterized builder (live-h2h4 L1 #3) --------------------------


def test_airline_gate_is_wired_to_both_structural_fast_rules() -> None:
    # Both of airline's structural boundaries (cancel_without_lookup,
    # unauthorized_rebooking) must be preventable pre-execution live, not
    # just cancel_without_lookup -- see runtime.langgraph_agent's
    # build_airline_agent_session docstring.
    toolkit = _CountingToolkit()
    llm = _ScriptedLLM(responses=[AIMessage(content="done")])
    session = build_airline_agent_session(
        trace_id="t-airline-gate-wiring", llm=llm, environment=_FakeEnvironment(tools=toolkit)
    )

    assert session.gate is not None
    gated_tools = {
        rule.gated_tool
        for rule in session.gate.instruments
        if isinstance(rule, RequireLookupBeforeCancel)
    }
    assert gated_tools == {"cancel_reservation", "update_reservation_flights"}


def test_build_retail_agent_session_wires_retail_fast_rules_without_network() -> None:
    toolkit = _CountingToolkit()
    llm = _ScriptedLLM(responses=[AIMessage(content="done")])

    session = build_retail_agent_session(
        trace_id="t-retail-wiring", llm=llm, environment=_FakeEnvironment(tools=toolkit)
    )

    assert session.gate is not None
    gated_tools = {
        rule.gated_tool
        for rule in session.gate.instruments
        if isinstance(rule, RequireLookupBeforeCancel)
    }
    assert gated_tools == {
        "cancel_pending_order",
        "return_delivered_order_items",
        "modify_pending_order_payment",
        "modify_user_address",
    }


def test_retail_agent_session_gate_blocks_cancel_without_prior_lookup() -> None:
    toolkit = _CountingToolkit()
    llm = _ScriptedLLM(
        responses=[
            AIMessage(
                content="",
                tool_calls=[_tool_call("cancel_pending_order", {"order_id": "#W1"}, "call-1")],
            ),
            AIMessage(content="done"),
        ]
    )
    session = build_retail_agent_session(
        trace_id="t-retail-block", llm=llm, environment=_FakeEnvironment(tools=toolkit)
    )

    _result, payloads = _run_to_completion(session, thread_id="retail-block")

    assert payloads[0]["auto_verdict"] == "block"
    assert toolkit.call_count("cancel_pending_order", order_id="#W1") == 0
    assert session.gate is not None
    assert session.gate.history == []


def test_retail_agent_session_builds_against_the_real_tau2_retail_environment() -> None:
    # No FIREWORKS_API_KEY / network needed: `llm` is a fake, and tau2's
    # retail environment loads from a local bundled JSON file, not a
    # network call -- confirms the retail tau2 env is cleanly available
    # (live-h2h4 L1 #3's stop-and-report condition does NOT apply).
    llm = _ScriptedLLM(responses=[AIMessage(content="done")])

    session = build_retail_agent_session(trace_id="t-retail-real-env", llm=llm)

    assert session.gate is not None
    gated_tools = {
        rule.gated_tool
        for rule in session.gate.instruments
        if isinstance(rule, RequireLookupBeforeCancel)
    }
    assert "cancel_pending_order" in gated_tools


def test_retail_tool_schemas_exposes_the_real_bound_retail_toolset() -> None:
    # The grounded adversary (conditions.fireworks_adversary) must ground
    # its attacks in the SAME tools the live retail agent binds -- so this
    # exposes the exact `openai_schema` list `_build_agent_session` binds.
    # No network / API key: tau2's retail env is a local JSON load.
    schemas = retail_tool_schemas()

    names = {schema["function"]["name"] for schema in schemas}
    # The 16 real retail tools (see runtime/langgraph_agent.py's binding).
    assert len(schemas) == 16
    assert {"cancel_pending_order", "modify_user_address", "get_user_details"} <= names
