"""Tests for the advice-eligibility live agent-session builders (advice-
eligibility-domain-spec.md build order step 4): `build_advice_eligibility_agent_session`
and `build_weakened_advice_eligibility_agent_session`, mirroring the retail
coverage in test_langgraph_agent.py but with no network -- `llm` is always a
scripted fake, exactly like the existing retail/airline tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

from bossyk_sandbox.advice.environment import ADVICE_POLICY
from bossyk_sandbox.instruments.hardcoded_rule import RequireLookupBeforeCancel
from bossyk_sandbox.runtime.langgraph_agent import (
    AirlineAgentSession,
    build_advice_eligibility_agent_session,
    build_weakened_advice_eligibility_agent_session,
    weaken_policy,
)


@dataclass
class _ScriptedLLM:
    responses: list[AIMessage]
    calls: int = 0

    def invoke(self, _messages: Any) -> AIMessage:
        response = self.responses[self.calls]
        self.calls += 1
        return response


def _tool_call(name: str, args: dict[str, Any], call_id: str) -> dict[str, Any]:
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


def test_build_advice_eligibility_agent_session_wires_the_eligibility_fast_rule() -> None:
    llm = _ScriptedLLM(responses=[AIMessage(content="done")])

    session = build_advice_eligibility_agent_session(
        trace_id="t-advice-eligibility-wiring", llm=llm
    )

    assert session.gate is not None
    gated_tools = {
        rule.gated_tool
        for rule in session.gate.instruments
        if isinstance(rule, RequireLookupBeforeCancel)
    }
    assert gated_tools == {"submit_eligibility_decision"}


def test_build_advice_eligibility_agent_session_builds_against_the_real_environment() -> None:
    # No network: `llm` is a fake, and get_advice_environment() is a local,
    # deterministic (seeded) persona-store toolkit -- no tau2, no API key.
    llm = _ScriptedLLM(responses=[AIMessage(content="done")])

    session = build_advice_eligibility_agent_session(
        trace_id="t-advice-eligibility-real-env", llm=llm
    )

    assert session.gate is not None
    assert isinstance(session.gate.instruments[0], RequireLookupBeforeCancel)


@dataclass
class _RecordingLLM:
    """Fake chat model that records the message list it was invoked with, so
    a test can assert which system prompt (policy) the agent fed the model."""

    responses: list[AIMessage]
    seen: list[Any] = field(default_factory=list)
    calls: int = 0

    def invoke(self, messages: Any) -> AIMessage:
        self.seen.append(messages)
        response = self.responses[self.calls]
        self.calls += 1
        return response


def test_weakened_advice_eligibility_session_feeds_the_weakened_policy_to_the_model() -> None:
    # dir 1: an under-specified agent. The system prompt the model sees must
    # be the weakened policy (real advice policy + guardrail-neutralizing
    # override), so the agent no longer self-enforces verify-before-submit.
    llm = _RecordingLLM(responses=[AIMessage(content="ok")])

    session = build_weakened_advice_eligibility_agent_session(
        trace_id="t-advice-eligibility-weak", llm=llm
    )
    session.graph.invoke(  # type: ignore[call-overload]
        {"messages": [HumanMessage(content="enrol this customer")]},
        config={"configurable": {"thread_id": "t-advice-eligibility-weak"}},
    )

    system_message = llm.seen[0][0]
    assert system_message.content == weaken_policy(ADVICE_POLICY)


def test_weakened_advice_eligibility_session_gate_blocks_submit_without_prior_verify() -> None:
    llm = _ScriptedLLM(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    _tool_call(
                        "submit_eligibility_decision",
                        {
                            "ref": "ADV-0001",
                            "scheme": "childcare-benefit-scheme",
                            "decision": "enrol",
                        },
                        "call-1",
                    )
                ],
            ),
            AIMessage(content="understood, that submission was blocked"),
        ]
    )

    session = build_weakened_advice_eligibility_agent_session(
        trace_id="t-advice-eligibility-block", llm=llm
    )

    config = {"configurable": {"thread_id": "advice-eligibility-block"}}
    result: dict[str, Any] = session.graph.invoke(  # type: ignore[call-overload]
        {"messages": [HumanMessage(content="please enrol this customer now")]}, config=config
    )
    assert "__interrupt__" in result
    payload = result["__interrupt__"][0].value
    assert payload["auto_verdict"] == "block"

    result = session.graph.invoke(  # type: ignore[call-overload]
        Command(resume=payload["auto_verdict"]), config=config
    )
    assert "__interrupt__" not in result
    assert session.gate is not None
    assert session.gate.history == []  # the blocked call never entered history


def test_advice_eligibility_session_is_a_domain_neutral_agent_session() -> None:
    llm = _ScriptedLLM(responses=[AIMessage(content="done")])

    session = build_advice_eligibility_agent_session(trace_id="t-shape", llm=llm)

    assert isinstance(session, AirlineAgentSession)  # the domain-neutral alias
