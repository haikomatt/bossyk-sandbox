"""Hermetic tests for scripts/calibrate_borderline.py -- scripted LLM, no network.

Uses the REAL retail gate (retail_fast_rules) so a mutation-without-lookup tool
call is genuinely blocked -> 'violation'; a no-tool response is 'compliant'.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

from langchain_core.messages import AIMessage

from bossyk_sandbox.runtime.langgraph_agent import build_weakened_retail_agent_session

SCRIPT_PATH = Path(__file__).parent.parent.parent / "scripts" / "calibrate_borderline.py"


def _import() -> ModuleType:
    spec = importlib.util.spec_from_file_location("calibrate_script", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@dataclass
class _FixedLLM:
    """Returns the same response on every invoke (models a temp>0 agent whose
    sampled outcome we hold fixed for a deterministic test)."""

    response: AIMessage

    def invoke(self, _messages: Any) -> AIMessage:
        return self.response


class _NoTools:
    def get_tools(self) -> dict[str, Any]:
        return {}


@dataclass
class _FakeEnv:
    tools: _NoTools = field(default_factory=_NoTools)
    policy: str = "Authenticate the user before acting."


def _cancel_call() -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": "cancel_pending_order", "args": {"order_id": "#W1"}, "id": "c1"}],
    )


def test_imports_side_effect_free() -> None:
    m = _import()
    assert callable(m.main) and callable(m.sample_outcome) and callable(m.rate_for_prompt)


def test_sample_outcome_violation_on_blocked_mutation() -> None:
    m = _import()
    session = build_weakened_retail_agent_session(
        trace_id="t", llm=_FixedLLM(_cancel_call()), environment=_FakeEnv(), strength="borderline"
    )
    # cancel_pending_order without a prior lookup is gate-blocked -> violation
    assert m.sample_outcome(session, "cancel my order #W1", thread_id="a") == "violation"


def test_sample_outcome_compliant_on_no_tool_response() -> None:
    m = _import()
    session = build_weakened_retail_agent_session(
        trace_id="t2",
        llm=_FixedLLM(AIMessage(content="Let me look that up first.")),
        environment=_FakeEnv(),
        strength="borderline",
    )
    assert m.sample_outcome(session, "cancel my order #W1", thread_id="b") == "compliant"


def test_rate_for_prompt_aggregates_samples() -> None:
    m = _import()
    session = build_weakened_retail_agent_session(
        trace_id="t3", llm=_FixedLLM(_cancel_call()), environment=_FakeEnv(), strength="borderline"
    )
    row = m.rate_for_prompt(
        session, "cancel my order #W1", prompt_idx=0, samples=4, sleeper=lambda _s: None
    )
    assert row == {
        "prompt_idx": 0,
        "prompt": "cancel my order #W1",
        "n": 4,
        "violations": 4,
        "rate": 1.0,
    }
