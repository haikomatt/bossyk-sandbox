"""Tests for `build_outreach_agent_session` (bossyk-sandbox slice 1 -> slice
2, P5).

Mirrors `build_retail_agent_session`'s wiring tests in
tests/unit/test_langgraph_agent.py (D1: same `_build_agent_session` seam,
`environment=` bypasses tau2 entirely). Local fakes rather than importing
another test module's, per repo convention (see tests/unit/test_live_session.py
and tests/unit/test_console_live.py, which each define their own fake
toolkit/environment rather than sharing test_langgraph_agent.py's).

Test-Integrity note: `outreach_fast_rules()` now returns `RequirePassedCheck`
(outcome-aware), not slice 1's `RequireLookupBeforeCancel` (precedence-only)
-- see test_outreach_fast_rules.py. The wiring/isinstance assertions here
were updated accordingly, and `_CountingToolkit` gained an optional
per-tool-name `results` map so a test can script a REAL check result (e.g.
`check_eligibility` -> `{"eligible": True}`), not just the generic "ok"
string -- required for `RequirePassedCheck`'s predicate to ever ALLOW.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

from bossyk_sandbox.instruments.hardcoded_rule import RequirePassedCheck
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
    # Per-tool-name canned result (e.g. {"check_eligibility": {"eligible": True}}),
    # so a test can script a real, outcome-aware check result -- falls back
    # to the generic "ok" string for any tool not listed here.
    results: dict[str, Any] = field(default_factory=dict)
    counts: dict[tuple[str, tuple[tuple[str, Any], ...]], int] = field(default_factory=dict)

    def get_tools(self) -> dict[str, Any]:
        return {}

    def use_tool(self, name: str, **kwargs: Any) -> Any:
        key = (name, tuple(sorted(kwargs.items())))
        self.counts[key] = self.counts.get(key, 0) + 1
        if name in self.raising_tools:
            raise RuntimeError("boom")
        return self.results.get(name, "ok")

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


def test_build_outreach_agent_session_wires_both_outcome_aware_fast_rules() -> None:
    toolkit = _CountingToolkit()
    llm = _ScriptedLLM(responses=[AIMessage(content="done")])

    session = build_outreach_agent_session(
        trace_id="t-outreach-wiring", llm=llm, environment=_FakeEnvironment(tools=toolkit)
    )

    assert session.gate is not None
    gated_tools = {
        rule.gated_tool for rule in session.gate.instruments if isinstance(rule, RequirePassedCheck)
    }
    assert gated_tools == {"book_survey", "place_call"}


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
    # Outcome-aware (P5): a passing prior check_eligibility must be OBSERVED
    # (a real {"eligible": True} result), not merely called -- see
    # _CountingToolkit.results.
    toolkit = _CountingToolkit(results={"check_eligibility": {"eligible": True}})
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
        rule.gated_tool for rule in session.gate.instruments if isinstance(rule, RequirePassedCheck)
    }
    assert "book_survey" in gated_tools


# --- boundary 1 live wiring (slice 2, P5, item 4): execute_node must record
# an ObservedAction with the tool's REAL result, not just the bare proposal,
# for RequirePassedCheck to work end-to-end. Not part of the reviewed RED
# commit (P5's RED phase covered the instrument logic via synthetic
# histories only) -- added here as the integration proof that the wiring
# itself (execute_node -> gate.record(ObservedAction(...))) is correct.


def test_boundary_1_live_session_blocks_place_call_after_an_observed_suppressed_check() -> None:
    # The agent DOES call check_suppression, the toolkit DOES return
    # on_tps=True, and the agent dials anyway -- precedence gating
    # (RequireLookupBeforeCancel) could never catch this; this is the case
    # RequirePassedCheck exists for.
    toolkit = _CountingToolkit(results={"check_suppression": {"on_tps": True, "opted_out": False}})
    llm = _ScriptedLLM(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    _tool_call("check_suppression", {"phone": "+441135550007"}, "call-1"),
                    _tool_call("place_call", {"phone": "+441135550007"}, "call-2"),
                ],
            ),
            AIMessage(content="done"),
        ]
    )
    session = build_outreach_agent_session(
        trace_id="t-outreach-boundary1-block", llm=llm, environment=_FakeEnvironment(tools=toolkit)
    )

    _result, payloads = _run_to_completion(session, thread_id="outreach-boundary1-block")

    assert len(payloads) == 2
    assert payloads[1]["auto_verdict"] == "block"
    assert toolkit.call_count("place_call", phone="+441135550007") == 0


def test_boundary_1_live_session_allows_place_call_after_an_observed_clean_check() -> None:
    toolkit = _CountingToolkit(results={"check_suppression": {"on_tps": False, "opted_out": False}})
    llm = _ScriptedLLM(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    _tool_call("check_suppression", {"phone": "+441135550001"}, "call-1"),
                    _tool_call("place_call", {"phone": "+441135550001"}, "call-2"),
                ],
            ),
            AIMessage(content="done"),
        ]
    )
    session = build_outreach_agent_session(
        trace_id="t-outreach-boundary1-allow", llm=llm, environment=_FakeEnvironment(tools=toolkit)
    )

    _result, payloads = _run_to_completion(session, thread_id="outreach-boundary1-allow")

    assert len(payloads) == 2
    assert payloads[1]["auto_verdict"] == "allow"
    assert toolkit.call_count("place_call", phone="+441135550001") == 1


def test_gate_history_stays_unwrapped_to_bare_proposed_actions_after_observed_execution() -> None:
    # Gate.history (the public property) must still return bare
    # ProposedAction, unwrapped, even though ObservedAction is now what
    # execute_node records internally -- this is the byte-identical
    # requirement for every existing caller that reads gate.history
    # (tests, live_replay.py's LiveRunResult.executed, the console).
    from bossyk_sandbox.instruments.base import ProposedAction

    toolkit = _CountingToolkit(results={"check_eligibility": {"eligible": True}})
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
        trace_id="t-outreach-history-unwrap", llm=llm, environment=_FakeEnvironment(tools=toolkit)
    )

    _run_to_completion(session, thread_id="outreach-history-unwrap")

    assert session.gate is not None
    assert session.gate.history == [
        ProposedAction("check_eligibility", {"prospect_id": "P-1"}),
        ProposedAction("book_survey", {"prospect_id": "P-1", "slot": "2026-08-03T10:00"}),
    ]
