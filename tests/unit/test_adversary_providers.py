from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from bossyk_sandbox.conditions.adversary_providers import _parse_anthropic_response


@dataclass
class _FakeTextBlock:
    text: str
    type: str = "text"


@dataclass
class _FakeRefusalResponse:
    """A fake Anthropic response whose `content` raises if read -- proves
    `_parse_anthropic_response` branches on `stop_reason` before ever
    touching `content` (a refusal's content can be empty or partial and
    must never be read as the payload)."""

    stop_reason: str = "refusal"
    stop_details: str = "cyber"

    @property
    def content(self) -> list[Any]:
        raise AssertionError("content must not be read when stop_reason is 'refusal'")


@dataclass
class _FakeTextResponse:
    stop_reason: str = "end_turn"
    content: list[_FakeTextBlock] = field(default_factory=lambda: [_FakeTextBlock(text="payload")])


def test_parse_anthropic_response_marks_a_refusal_stop_reason_as_refused() -> None:
    result = _parse_anthropic_response(_FakeRefusalResponse())

    assert result.refused is True
    assert result.text == ""
    assert result.detail == "cyber"


def test_parse_anthropic_response_returns_the_text_blocks_text_when_not_refused() -> None:
    result = _parse_anthropic_response(_FakeTextResponse())

    assert result.refused is False
    assert result.text == "payload"


def test_parse_anthropic_response_checks_stop_reason_before_reading_content() -> None:
    # _FakeRefusalResponse.content raises on access -- this only passes if
    # the parser inspects stop_reason first and never touches content.
    result = _parse_anthropic_response(_FakeRefusalResponse())

    assert result.refused is True
