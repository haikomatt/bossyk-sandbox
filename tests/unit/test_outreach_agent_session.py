"""RED-phase tests for `build_outreach_agent_session` (bossyk-sandbox slice 1).

Mirrors `build_retail_agent_session`'s wiring tests in
tests/unit/test_langgraph_agent.py (D1: same `_build_agent_session` seam,
`environment=` bypasses tau2 entirely). Local fakes rather than importing
another test module's, per repo convention (see tests/unit/test_live_session.py
and tests/unit/test_console_live.py, which each define their own fake
toolkit/environment rather than sharing test_langgraph_agent.py's).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

from bossyk_sandbox.instruments.hardcoded_rule import RequireLookupBeforeCancel
from bossyk_sandbox.runtime.langgraph_agent import build_outreach_agent_session


@dataclass
class _ScriptedLLM:
    responses: list[AIMessage]
    calls: int = 0

    def invoke(self, _messages: Any) -> AIMessage:
        response = self.responses[self.calls]
        self.calls += 1
        return response


@dataclass
class _CountingToolkit:
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
    tools: _CountingToolkit
    policy: str = "test outreach policy"


def _tool_call(name: str, args: dict[str, Any], call_id: str) -> dict[str, Any]:
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


def _run_to_completion(
    session: Any,
    *,
    thread_id: str,
    resume_values: dict[str, str] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config = {"configurable": {"thread_id": thread_id}}
    resume_values = resume_values or {}
    payloads: list[dict[str, Any]] = []

    result: dict[str, Any] = session.graph.invoke(
        {"messages": [HumanMessage(content="please book a free survey")]}, config=config
    )
    while "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        payloads.append(payload)
        resume = resume_values.get(payload["tool_call_id"], payload["auto_verdict"])
        result = session.graph.invoke(Command(resume=resume), config=config)
    return result, payloads


def test_build_outreach_agent_session_wires_a_book_survey_fast_rule_without_network() -> None:
    toolkit = _CountingToolkit()
    llm = _ScriptedLLM(responses=[AIMessage(content="done")])

    session = build_outreach_agent_session(
        trace_id="t-outreach-wiring", llm=llm, environment=_FakeEnvironment(tools=toolkit)
    )

    assert session.gate is not None
    gated_tools = {
        rule.gated_tool
        for rule in session.gate.instruments
        if isinstance(rule, RequireLookupBeforeCancel)
    }
    assert "book_survey" in gated_tools


def test_outreach_agent_session_gate_blocks_booking_without_prior_eligibility_check() -> None:
    toolkit = _CountingToolkit()
    llm = _ScriptedLLM(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    _tool_call(
                        "book_survey",
                        {"prospect_id": "P-1", "slot": "2026-08-03T10:00"},
                        "call-1",
                    )
                ],
            ),
            AIMessage(content="done"),
        ]
    )
    session = build_outreach_agent_session(
        trace_id="t-outreach-block", llm=llm, environment=_FakeEnvironment(tools=toolkit)
    )

    _result, payloads = _run_to_completion(session, thread_id="outreach-block")

    assert payloads[0]["auto_verdict"] == "block"
    assert toolkit.call_count("book_survey", prospect_id="P-1", slot="2026-08-03T10:00") == 0
    assert session.gate is not None
    assert session.gate.history == []


def test_outreach_agent_session_gate_allows_booking_after_prior_eligibility_check() -> None:
    toolkit = _CountingToolkit()
    llm = _ScriptedLLM(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    _tool_call("check_eligibility", {"prospect_id": "P-1"}, "call-1"),
                    _tool_call(
                        "book_survey",
                        {"prospect_id": "P-1", "slot": "2026-08-03T10:00"},
                        "call-2",
                    ),
                ],
            ),
            AIMessage(content="done"),
        ]
    )
    session = build_outreach_agent_session(
        trace_id="t-outreach-allow", llm=llm, environment=_FakeEnvironment(tools=toolkit)
    )

    _result, payloads = _run_to_completion(session, thread_id="outreach-allow")

    assert len(payloads) == 2
    assert payloads[1]["auto_verdict"] == "allow"
    assert toolkit.call_count("book_survey", prospect_id="P-1", slot="2026-08-03T10:00") == 1


def test_build_outreach_agent_session_builds_against_the_real_outreach_environment() -> None:
    # No FIREWORKS_API_KEY / network needed: `llm` is a fake, and the real
    # outreach environment (get_outreach_environment) must be a local,
    # deterministic fixture-DB load -- not a tau2 or network call.
    llm = _ScriptedLLM(responses=[AIMessage(content="done")])

    session = build_outreach_agent_session(trace_id="t-outreach-real-env", llm=llm)

    assert session.gate is not None
    gated_tools = {
        rule.gated_tool
        for rule in session.gate.instruments
        if isinstance(rule, RequireLookupBeforeCancel)
    }
    assert "book_survey" in gated_tools
