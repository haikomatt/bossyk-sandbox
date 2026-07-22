from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from bossyk_sandbox.conditions.adversary import TokenUsage
from bossyk_sandbox.conditions.adversary_providers import (
    OpenAICompatibleChatClient,
    _parse_anthropic_response,
)


@dataclass
class _FakeUsage:
    input_tokens: int
    output_tokens: int


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
    # A refusal still burns tokens (the input + the model's thinking) -- the
    # ledger must capture that cost, so the fake carries usage too.
    usage: _FakeUsage = field(default_factory=lambda: _FakeUsage(input_tokens=50, output_tokens=5))

    @property
    def content(self) -> list[Any]:
        raise AssertionError("content must not be read when stop_reason is 'refusal'")


@dataclass
class _FakeTextResponse:
    stop_reason: str = "end_turn"
    content: list[_FakeTextBlock] = field(default_factory=lambda: [_FakeTextBlock(text="payload")])
    usage: _FakeUsage = field(default_factory=lambda: _FakeUsage(input_tokens=12, output_tokens=34))


@dataclass
class _FakeMessage:
    """Stand-in for a langchain `AIMessage` -- `.content` plus the
    `.usage_metadata` dict langchain populates from an OpenAI-compatible
    provider's token counts (None when the provider returns no usage)."""

    content: str
    usage_metadata: dict[str, int] | None


@dataclass
class _FakeLLM:
    message: _FakeMessage

    def invoke(self, _messages: Any) -> _FakeMessage:
        return self.message


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


def test_parse_anthropic_response_records_token_usage() -> None:
    result = _parse_anthropic_response(_FakeTextResponse())

    assert result.usage == TokenUsage(input_tokens=12, output_tokens=34)


def test_parse_anthropic_response_records_usage_even_on_refusal() -> None:
    # A refused generation is not free -- its tokens must still hit the ledger.
    result = _parse_anthropic_response(_FakeRefusalResponse())

    assert result.refused is True
    assert result.usage == TokenUsage(input_tokens=50, output_tokens=5)


def test_openai_compatible_client_records_usage_from_usage_metadata() -> None:
    client = OpenAICompatibleChatClient(
        llm=_FakeLLM(_FakeMessage("payload", {"input_tokens": 7, "output_tokens": 9}))
    )

    result = client.complete("system", "user")

    assert result.text == "payload"
    assert result.usage == TokenUsage(input_tokens=7, output_tokens=9)


def test_openai_compatible_client_tolerates_missing_usage_metadata() -> None:
    # Some providers omit usage; the client must degrade to zero, not crash.
    client = OpenAICompatibleChatClient(llm=_FakeLLM(_FakeMessage("payload", None)))

    assert client.complete("system", "user").usage == TokenUsage()
