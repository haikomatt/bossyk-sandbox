"""Hermetic tests for interp.prompt_render + interp.decisions."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from bossyk_sandbox.interp.decisions import build_decisions, decisions_to_json
from bossyk_sandbox.interp.prompt_render import render_prompt


def test_render_prompt_transcribes_roles_and_tool_calls() -> None:
    messages = [
        SystemMessage(content="policy: be good"),
        HumanMessage(content="cancel my order"),
        AIMessage(content="", tool_calls=[{"name": "cancel", "args": {"id": "W1"}, "id": "c1"}]),
        ToolMessage(content="BLOCKED", tool_call_id="c1"),
    ]
    text = render_prompt(messages)
    assert "system: policy: be good" in text
    assert "human: cancel my order" in text
    assert "ai:  [tool_calls: cancel({'id': 'W1'})]" in text
    assert "tool: BLOCKED" in text


def test_render_prompt_empty() -> None:
    assert render_prompt([]) == ""


def test_build_decisions_labels_blocked_turns_as_violations() -> None:
    prompts = ["p0", "p1", "p2"]
    items = build_decisions(prompts, {1})
    assert [it.step_id for it in items] == ["turn-0", "turn-1", "turn-2"]
    assert [it.is_violation for it in items] == [False, True, False]
    assert all(it.is_error is None for it in items)


def test_build_decisions_with_error_labels() -> None:
    items = build_decisions(["p0", "p1"], set(), errors=[True, False])
    assert [it.is_error for it in items] == [True, False]


def test_build_decisions_rejects_misaligned_errors() -> None:
    with pytest.raises(ValueError, match="errors length"):
        build_decisions(["p0", "p1"], set(), errors=[True])


def test_decisions_to_json_round_trips_schema() -> None:
    items = build_decisions(["p0", "p1"], {0}, errors=[False, True])
    rows = decisions_to_json(items)
    assert rows[0] == {"step_id": "turn-0", "prompt": "p0", "is_violation": True, "is_error": False}
    assert rows[1]["is_error"] is True


def test_decisions_to_json_omits_absent_error() -> None:
    rows = decisions_to_json(build_decisions(["p0"], set()))
    assert "is_error" not in rows[0]
