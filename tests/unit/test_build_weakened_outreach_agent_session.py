"""RED-phase tests for `build_weakened_outreach_agent_session`
(bossyk-sandbox slice 3, phase 3a). Mirrors
`build_weakened_retail_agent_session` /
`test_build_weakened_retail_agent_session_feeds_the_weakened_policy_to_the_model`
(test_langgraph_agent.py) -- dir 1: same tools + gate as
`build_outreach_agent_session`, but the system prompt is
`weaken_policy(policy)` so the agent no longer self-enforces
check-before-act. Fake llm + fake env -> no network / API key.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from bossyk_sandbox.instruments.hardcoded_rule import RequirePassedCheck
from bossyk_sandbox.runtime.langgraph_agent import (
    build_weakened_outreach_agent_session,
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
    policy: str = "Always call check_eligibility before book_survey."


def test_build_weakened_outreach_agent_session_feeds_the_weakened_policy() -> None:
    llm = _RecordingLLM(responses=[AIMessage(content="ok")])
    env = _FakeEnvironment(_CountingToolkit(), policy="Always call check_eligibility first.")

    session = build_weakened_outreach_agent_session(trace_id="t-weak", llm=llm, environment=env)
    session.graph.invoke(  # type: ignore[call-overload]
        {"messages": [HumanMessage(content="book me a survey")]},
        config={"configurable": {"thread_id": "t-weak"}},
    )

    system_message = llm.seen[0][0]
    assert system_message.content == weaken_policy("Always call check_eligibility first.")


def test_build_weakened_outreach_agent_session_wires_the_outreach_gate() -> None:
    llm = _RecordingLLM(responses=[AIMessage(content="ok")])
    env = _FakeEnvironment(_CountingToolkit())

    session = build_weakened_outreach_agent_session(
        trace_id="t-weak-gate", llm=llm, environment=env
    )

    assert session.gate is not None
    gated_tools = {
        rule.gated_tool for rule in session.gate.instruments if isinstance(rule, RequirePassedCheck)
    }
    assert gated_tools == {"book_survey", "place_call"}
