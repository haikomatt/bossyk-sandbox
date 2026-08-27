"""Hermetic tests for bossyk_sandbox.detector.judge -- fake ChatModel, no network."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
from langchain_core.messages import AIMessage

from bossyk_sandbox.detector.judge import (
    JudgeVerdict,
    build_judge_user_prompt,
    judge_decision,
    judge_decisions,
    parse_judge_verdict,
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
    v = parse_judge_verdict('{"p_violation": 0.9, "reason": "skipped verification"}')
    assert v == JudgeVerdict(p_violation=0.9, reason="skipped verification")


def test_parse_accepts_integer_p_violation() -> None:
    v = parse_judge_verdict('{"p_violation": 1, "reason": "clear violation"}')
    assert v.p_violation == 1.0


def test_parse_tolerates_code_fence_and_prose() -> None:
    raw = 'My assessment:\n```json\n{"p_violation": 0.1, "reason": "compliant"}\n```\nDone.'
    v = parse_judge_verdict(raw)
    assert v.p_violation == 0.1


def test_parse_rejects_no_json() -> None:
    with pytest.raises(ValueError, match="no JSON object"):
        parse_judge_verdict("looks fine to me")


def test_parse_rejects_missing_key() -> None:
    with pytest.raises(ValueError, match="missing 'p_violation'"):
        parse_judge_verdict('{"reason": "no verdict"}')


def test_parse_rejects_non_numeric_p_violation() -> None:
    with pytest.raises(ValueError, match="not numeric"):
        parse_judge_verdict('{"p_violation": "high", "reason": "x"}')


def test_parse_rejects_boolean_p_violation() -> None:
    # bool is a subclass of int in Python -- must be explicitly rejected.
    with pytest.raises(ValueError, match="not numeric"):
        parse_judge_verdict('{"p_violation": true, "reason": "x"}')


def test_parse_rejects_out_of_range_p_violation() -> None:
    with pytest.raises(ValueError, match=r"out of \[0, 1\]"):
        parse_judge_verdict('{"p_violation": 1.5, "reason": "x"}')


def test_build_judge_user_prompt_includes_policy_context_and_action() -> None:
    prompt = build_judge_user_prompt(
        "policy_id: retail-v1", "customer wants a refund", "cancel(W1)"
    )
    assert "policy_id: retail-v1" in prompt
    assert "customer wants a refund" in prompt
    assert "cancel(W1)" in prompt


def test_judge_decision_invokes_llm_and_parses() -> None:
    llm = _ScriptedJudge(contents=['{"p_violation": 0.8, "reason": "no lookup"}'])
    verdict = judge_decision("policy", "context", "action", llm=llm)
    assert verdict.p_violation == 0.8
    assert llm.calls == 1


def test_judge_decision_rejects_non_string_content() -> None:
    class _NonStringJudge:
        def invoke(self, _messages: list[Any]) -> AIMessage:
            return AIMessage(content=[{"type": "text", "text": "nope"}])

    with pytest.raises(ValueError, match="non-string"):
        judge_decision("policy", "context", "action", llm=_NonStringJudge())


def test_judge_decisions_labels_each_item_in_order() -> None:
    llm = _ScriptedJudge(
        contents=[
            '{"p_violation": 0.9, "reason": "a"}',
            '{"p_violation": 0.05, "reason": "b"}',
        ]
    )
    items = [("policy", "ctx0", "act0"), ("policy", "ctx1", "act1")]
    verdicts = judge_decisions(items, llm=llm, sleeper=lambda _s: None)
    assert verdicts[0] is not None and verdicts[0].p_violation == 0.9
    assert verdicts[1] is not None and verdicts[1].p_violation == 0.05


def test_judge_decisions_retries_transient_errors_then_succeeds() -> None:
    llm = _ScriptedJudge(
        contents=[
            RuntimeError("503 Service Overloaded"),
            '{"p_violation": 0.3, "reason": "retried ok"}',
        ]
    )
    verdicts = judge_decisions([("p", "c", "a")], llm=llm, sleeper=lambda _s: None)
    assert verdicts[0] is not None
    assert verdicts[0].p_violation == 0.3
    assert llm.calls == 2


def test_judge_decisions_gives_up_on_non_transient_error() -> None:
    llm = _ScriptedJudge(contents=[RuntimeError("401 Unauthorized")])
    verdicts = judge_decisions([("p", "c", "a")], llm=llm, sleeper=lambda _s: None)
    assert verdicts[0] is None
    assert llm.calls == 1  # no retry for a non-transient error


def test_judge_decisions_leaves_persistently_unparseable_item_unjudged() -> None:
    llm = _ScriptedJudge(contents=["not json at all"] * 4)
    verdicts = judge_decisions([("p", "c", "a")], llm=llm, max_attempts=4, sleeper=lambda _s: None)
    assert verdicts[0] is None
    assert llm.calls == 1  # a parse ValueError is not in the transient taxonomy -> no retry
