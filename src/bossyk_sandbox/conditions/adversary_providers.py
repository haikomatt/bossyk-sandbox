from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from bossyk_sandbox.conditions.adversary import (
    ChatResult,
    usage_from_anthropic,
    usage_from_langchain,
)

if TYPE_CHECKING:
    # Lazily imported for real at call time in `build_anthropic_client` --
    # this module-level import only exists for mypy; it never runs at
    # runtime (mirrors the drift.py/policy.py build_* gated-import pattern).
    import anthropic

DEFAULT_ANTHROPIC_MODEL = "claude-fable-5"

# Exact beta flag required to opt in to Claude Fable 5's server-side
# refusal fallback -- do not "correct" the date suffix, see
# AnthropicChatClient.complete.
ANTHROPIC_FALLBACK_BETA = "server-side-fallback-2026-06-01"


@dataclass
class OpenAICompatibleChatClient:
    """Real `ChatClient` for any OpenAI-compatible provider endpoint
    (Fireworks and friends) -- generalizes `FireworksChatClient`
    (fireworks_adversary.py) to a provider-agnostic client. OpenAI-compatible
    providers surface no reliable refusal stop-reason, so `complete` always
    reports `refused=False`."""

    llm: ChatOpenAI

    def complete(self, system_prompt: str, user_prompt: str) -> ChatResult:
        response = self.llm.invoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        )
        return ChatResult(
            text=str(response.content), refused=False, usage=usage_from_langchain(response)
        )


def build_openai_compatible_client(
    *, base_url: str, api_key: str, model: str, temperature: float = 1.0
) -> OpenAICompatibleChatClient:
    """Lazy factory -- constructs `ChatOpenAI` only when called, no network
    call at construction time (the network call happens in `complete`),
    mirroring `build_fireworks_adversary`'s client construction
    (fireworks_adversary.py)."""
    llm = ChatOpenAI(
        model=model,
        base_url=base_url,
        api_key=SecretStr(api_key),
        temperature=temperature,
    )
    return OpenAICompatibleChatClient(llm=llm)


def _parse_anthropic_response(response: Any) -> ChatResult:
    """Pure, testable parser for an Anthropic Messages API response.
    `Any`: the SDK response type isn't imported in this import-light
    module, and the concrete class differs between `messages.create` and
    `beta.messages.create` -- both shapes only need `stop_reason`,
    `stop_details`, and `content` duck-typed here.

    Checks `stop_reason` BEFORE reading `content` -- a refusal's `content`
    can be empty (declined before any output) or a partial response
    (declined mid-stream), and must never be treated as the payload."""
    usage = usage_from_anthropic(response)
    if response.stop_reason == "refusal":
        return ChatResult(
            text="",
            refused=True,
            detail=str(getattr(response, "stop_details", "") or "refusal"),
            usage=usage,
        )
    text = next(
        (block.text for block in response.content if getattr(block, "type", None) == "text"),
        "",
    )
    return ChatResult(text=text, refused=False, usage=usage)


@dataclass
class AnthropicChatClient:
    """Real `ChatClient` backed by the official `anthropic` SDK. Handles
    Claude Fable 5's non-standard API surface -- see `complete` below and
    `build_anthropic_client`'s docstring for the exact usage constraints."""

    client: anthropic.Anthropic
    model: str = DEFAULT_ANTHROPIC_MODEL
    max_tokens: int = 1024
    effort: str | None = None
    fallback_model: str | None = None

    def complete(self, system_prompt: str, user_prompt: str) -> ChatResult:
        # Fable 5's thinking is always-on -- the `thinking` param is
        # intentionally never sent (sending it errors 400). Depth is
        # controlled via output_config.effort instead.
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        }
        if self.effort is not None:
            kwargs["output_config"] = {"effort": self.effort}
        if self.fallback_model is None:
            response = self.client.messages.create(**kwargs)
        else:
            response = self.client.beta.messages.create(
                **kwargs,
                betas=[ANTHROPIC_FALLBACK_BETA],
                fallbacks=[{"model": self.fallback_model}],
            )
        return _parse_anthropic_response(response)


def build_anthropic_client(
    *,
    model: str = DEFAULT_ANTHROPIC_MODEL,
    max_tokens: int = 1024,
    effort: str | None = None,
    fallback_model: str | None = None,
) -> AnthropicChatClient:
    """Real adversary client path: lazily imports `anthropic` and
    constructs `anthropic.Anthropic()` only when called -- no import-time
    SDK construction, so unit tests never need network access. Requires
    ANTHROPIC_API_KEY (read from the environment by the SDK itself) --
    gated here first for a clear error, mirroring
    `build_fireworks_adversary`'s FIREWORKS_API_KEY gate."""
    import anthropic

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY is required to run the live Anthropic adversary "
            "(set it in .env, matching build_fireworks_adversary's convention)."
        )
    return AnthropicChatClient(
        client=anthropic.Anthropic(),
        model=model,
        max_tokens=max_tokens,
        effort=effort,
        fallback_model=fallback_model,
    )
