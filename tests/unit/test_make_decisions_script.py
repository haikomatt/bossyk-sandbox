"""Hermetic tests for scripts/make_decisions.py.

Imports the script by path and drives `drive_session` with a scripted LLM (no
network). Never calls main().
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from langchain_core.messages import AIMessage

from bossyk_sandbox.runtime.langgraph_agent import build_airline_agent_session

SCRIPT_PATH = Path(__file__).parent.parent.parent / "scripts" / "make_decisions.py"


def _import_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("make_decisions_script", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass
class _ScriptedLLM:
    responses: list[AIMessage]
    calls: int = 0

    def invoke(self, _messages: Any) -> AIMessage:
        r = self.responses[self.calls]
        self.calls += 1
        return r


@dataclass
class _FakeEnv:
    tools: Any
    policy: str = "be compliant"


class _NoTools:
    def get_tools(self) -> dict[str, Any]:
        return {}


def test_imports_side_effect_free_and_defines_main() -> None:
    module = _import_script()
    assert callable(module.main)
    assert callable(module.drive_session)
    assert isinstance(module.DEFAULT_PROMPTS, list) and module.DEFAULT_PROMPTS


def test_load_user_prompts() -> None:
    module = _import_script()
    assert module.load_user_prompts(["a", "b"]) == ["a", "b"]


def test_drive_session_builds_decisions_from_captured_prompts() -> None:
    module = _import_script()
    # A single no-tool turn per user prompt -> one decision per prompt, all
    # compliant (no blocked tool calls), each carrying the rendered context.
    llm = _ScriptedLLM(responses=[AIMessage(content="ok"), AIMessage(content="ok")])
    session = build_airline_agent_session(
        trace_id="t-decisions",
        llm=llm,
        environment=_FakeEnv(tools=_NoTools(), policy="POLICY-X"),
        capture_prompts=True,
    )
    items = module.drive_session(session, ["hello", "again"])
    assert len(items) == 2
    assert [it.step_id for it in items] == ["turn-0", "turn-1"]
    assert all(it.is_violation is False for it in items)
    assert "POLICY-X" in items[0].prompt


class _FlakyLLM:
    """Raises a transient (503) error on the first N calls, then succeeds."""

    def __init__(self, fail_times: int) -> None:
        self.calls = 0
        self.fail_times = fail_times

    def invoke(self, _messages: Any) -> AIMessage:
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError("Error code: 503 - {'message': 'service overloaded'}")
        return AIMessage(content="ok")


def test_drive_session_retries_transient_503_then_succeeds() -> None:
    module = _import_script()
    session = build_airline_agent_session(
        trace_id="t-flaky",
        llm=_FlakyLLM(fail_times=2),  # fails twice, succeeds on the 3rd attempt
        environment=_FakeEnv(tools=_NoTools(), policy="P"),
        capture_prompts=True,
    )
    items = module.drive_session(session, ["hello"], sleeper=lambda _s: None)
    assert len(items) == 1
    assert items[0].is_violation is False
    # partial captures from the two failed attempts were discarded, not accumulated
    assert len(session.agent_prompts) == 1


def test_drive_session_skips_prompt_that_never_recovers() -> None:
    module = _import_script()
    session = build_airline_agent_session(
        trace_id="t-persistent-fail",
        llm=_FlakyLLM(fail_times=99),  # never succeeds within max_attempts
        environment=_FakeEnv(tools=_NoTools(), policy="P"),
        capture_prompts=True,
    )
    items = module.drive_session(session, ["hello"], max_attempts=3, sleeper=lambda _s: None)
    assert items == []  # skipped, run did not crash
    assert session.agent_prompts == []  # no partial turns left behind
