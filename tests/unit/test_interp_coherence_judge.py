"""Hermetic tests for interp.coherence_judge -- fake ChatModel, no network."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
from langchain_core.messages import AIMessage

from bossyk_sandbox.interp.coherence_judge import (
    CoherenceVerdict,
    build_judge_user_prompt,
    judge_action_coherence,
    judge_items,
    parse_coherence_verdict,
)


@dataclass
class _ScriptedJudge:
    """Fake ChatModel: returns the next canned content, or raises the next
    scripted exception, per invoke."""

    contents: list[str | Exception]
    calls: int = 0

    def invoke(self, _messages: list[Any]) -> AIMessage:
        item = self.contents[self.calls]
        self.calls += 1
        if isinstance(item, Exception):
            raise item
        return AIMessage(content=item)


def test_parse_plain_json() -> None:
    v = parse_coherence_verdict('{"coherent": true, "reason": "matches request"}')
    assert v == CoherenceVerdict(is_error=False, reason="matches request")


def test_parse_incoherent_becomes_error() -> None:
    v = parse_coherence_verdict('{"coherent": false, "reason": "wrong tool"}')
    assert v.is_error is True and v.reason == "wrong tool"


def test_parse_tolerates_code_fence_and_prose() -> None:
    raw = 'Here is my rating:\n```json\n{"coherent": false, "reason": "garbled"}\n```\nDone.'
    assert parse_coherence_verdict(raw).is_error is True


def test_parse_rejects_no_json() -> None:
    with pytest.raises(ValueError, match="no JSON object"):
        parse_coherence_verdict("the action seems fine to me")


def test_parse_rejects_missing_coherent_key() -> None:
    with pytest.raises(ValueError, match="missing 'coherent'"):
        parse_coherence_verdict('{"reason": "no verdict"}')


def test_parse_rejects_non_boolean_coherent() -> None:
    with pytest.raises(ValueError, match="not a boolean"):
        parse_coherence_verdict('{"coherent": "yes"}')


def test_build_judge_user_prompt_contains_context_and_action() -> None:
    p = build_judge_user_prompt("SEEN CONTEXT", "DID THIS")
    assert "SEEN CONTEXT" in p and "DID THIS" in p


def test_judge_action_coherence_end_to_end() -> None:
    llm = _ScriptedJudge(contents=['{"coherent": true, "reason": "ok"}'])
    v = judge_action_coherence("ctx", "action", llm=llm)
    assert v.is_error is False


def test_judge_action_coherence_rejects_non_string_content() -> None:
    @dataclass
    class _BadJudge:
        def invoke(self, _messages: list[Any]) -> AIMessage:
            return AIMessage(content=[{"type": "text"}])  # non-string content

    with pytest.raises(ValueError, match="non-string"):
        judge_action_coherence("ctx", "action", llm=_BadJudge())


def test_judge_items_maps_verdicts_in_order() -> None:
    llm = _ScriptedJudge(
        contents=['{"coherent": true, "reason": "a"}', '{"coherent": false, "reason": "b"}']
    )
    verdicts = judge_items([("c0", "a0"), ("c1", "a1")], llm=llm, sleeper=lambda _s: None)
    assert [v.is_error for v in verdicts if v is not None] == [False, True]


def test_judge_items_retries_transient_then_succeeds() -> None:
    llm = _ScriptedJudge(
        contents=[
            RuntimeError("Error code: 503 - service overloaded"),
            '{"coherent": true, "reason": "recovered"}',
        ]
    )
    verdicts = judge_items([("c", "a")], llm=llm, sleeper=lambda _s: None)
    assert verdicts[0] is not None and verdicts[0].is_error is False


def test_judge_items_unparseable_yields_none_not_a_fabricated_label() -> None:
    llm = _ScriptedJudge(contents=["not json at all"])
    verdicts = judge_items([("c", "a")], llm=llm, sleeper=lambda _s: None)
    assert verdicts == [None]


def test_judge_items_persistent_transient_yields_none() -> None:
    llm = _ScriptedJudge(contents=[RuntimeError("503 overloaded")] * 4)
    verdicts = judge_items([("c", "a")], llm=llm, max_attempts=4, sleeper=lambda _s: None)
    assert verdicts == [None]


def test_judge_items_retries_request_timed_out() -> None:
    # "Request timed out." (no "timeout") must be treated as transient + retried
    llm = _ScriptedJudge(
        contents=[RuntimeError("Request timed out."), '{"coherent": true, "reason": "ok"}']
    )
    verdicts = judge_items([("c", "a")], llm=llm, sleeper=lambda _s: None)
    assert verdicts[0] is not None and verdicts[0].is_error is False
