"""RED-phase tests for `build_weakened_airline_agent_session`
(phase-detector-training-step2-datagen.md Part A item 1: the scaled
generation driver needs a weakened builder for every domain it drives,
airline included -- only retail and advice-eligibility had one before this).
Mirrors `build_weakened_retail_agent_session` /
`test_build_weakened_retail_agent_session_feeds_the_weakened_policy_to_the_model`
(test_langgraph_agent.py): same tools + gate as `build_airline_agent_session`,
but the system prompt is `weaken_policy(policy)` so the agent no longer
self-enforces lookup-before-mutate. Fake llm + fake env -> no network / API key.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from bossyk_sandbox.instruments.hardcoded_rule import RequireLookupBeforeCancel
from bossyk_sandbox.runtime.langgraph_agent import (
    build_weakened_airline_agent_session,
    weaken_policy,
)


@dataclass
class _RecordingLLM:
    responses: list[AIMessage]
    seen: list[Any] = field(default_factory=list)
    calls: int = 0

    def invoke(self, messages: Any) -> AIMessage:
        self.seen.append(messages)
        response = self.responses[self.calls]
        self.calls += 1
        return response


@dataclass
class _CountingToolkit:
    counts: dict[tuple[str, tuple[tuple[str, Any], ...]], int] = field(default_factory=dict)

    def get_tools(self) -> dict[str, Any]:
        return {}

    def use_tool(self, name: str, **kwargs: Any) -> str:
        key = (name, tuple(sorted(kwargs.items())))
        self.counts[key] = self.counts.get(key, 0) + 1
        return "ok"


@dataclass
class _FakeEnvironment:
    tools: _CountingToolkit
    policy: str = "Always look up a reservation before mutating it."


def test_build_weakened_airline_agent_session_feeds_the_weakened_policy() -> None:
    llm = _RecordingLLM(responses=[AIMessage(content="ok")])
    env = _FakeEnvironment(_CountingToolkit(), policy="Always look up before cancelling.")

    session = build_weakened_airline_agent_session(trace_id="t-weak-air", llm=llm, environment=env)
    session.graph.invoke(  # type: ignore[call-overload]
        {"messages": [HumanMessage(content="cancel my reservation")]},
        config={"configurable": {"thread_id": "t-weak-air"}},
    )

    system_message = llm.seen[0][0]
    assert system_message.content == weaken_policy("Always look up before cancelling.")


def test_build_weakened_airline_agent_session_accepts_strength_and_temperature() -> None:
    llm = _RecordingLLM(responses=[AIMessage(content="ok")])
    env = _FakeEnvironment(_CountingToolkit(), policy="Always look up before cancelling.")

    session = build_weakened_airline_agent_session(
        trace_id="t-weak-air-aggr", llm=llm, environment=env, strength="aggressive", temperature=0.8
    )
    session.graph.invoke(  # type: ignore[call-overload]
        {"messages": [HumanMessage(content="cancel my reservation")]},
        config={"configurable": {"thread_id": "t-weak-air-aggr"}},
    )
    assert llm.seen[0][0].content == weaken_policy(
        "Always look up before cancelling.", strength="aggressive"
    )


def test_build_weakened_airline_agent_session_wires_the_airline_gate() -> None:
    llm = _RecordingLLM(responses=[AIMessage(content="ok")])
    env = _FakeEnvironment(_CountingToolkit())

    session = build_weakened_airline_agent_session(
        trace_id="t-weak-air-gate", llm=llm, environment=env
    )

    assert session.gate is not None
    gated_tools = {
        rule.gated_tool
        for rule in session.gate.instruments
        if isinstance(rule, RequireLookupBeforeCancel)
    }
    assert "cancel_reservation" in gated_tools


def test_build_weakened_airline_agent_session_capture_prompts_records_context() -> None:
    llm = _RecordingLLM(responses=[AIMessage(content="ok")])
    env = _FakeEnvironment(_CountingToolkit(), policy="Always look up before cancelling.")

    session = build_weakened_airline_agent_session(
        trace_id="t-weak-air-capture", llm=llm, environment=env, capture_prompts=True
    )
    session.graph.invoke(  # type: ignore[call-overload]
        {"messages": [HumanMessage(content="cancel my reservation")]},
        config={"configurable": {"thread_id": "t-weak-air-capture"}},
    )
    assert len(session.agent_prompts) == 1
    assert weaken_policy("Always look up before cancelling.") in session.agent_prompts[0]
