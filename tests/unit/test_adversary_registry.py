from __future__ import annotations

import pytest

from bossyk_sandbox.conditions.adversary_registry import (
    ADVERSARY_MODELS,
    build_adversary,
    required_key_env,
)


def test_build_adversary_raises_key_error_for_an_unregistered_model_name() -> None:
    with pytest.raises(KeyError):
        build_adversary("telecom")


def test_fireworks_deepseek_spec_has_the_expected_provider_key_env_and_model_id() -> None:
    spec = ADVERSARY_MODELS["fireworks-deepseek"]

    assert spec.provider == "openai_compatible"
    assert spec.key_env == "FIREWORKS_API_KEY"
    assert spec.model_id == "accounts/fireworks/models/deepseek-v4-pro"


def test_anthropic_fable_spec_has_the_expected_provider_key_env_and_model_id() -> None:
    spec = ADVERSARY_MODELS["anthropic-fable"]

    assert spec.provider == "anthropic"
    assert spec.key_env == "ANTHROPIC_API_KEY"
    assert spec.model_id == "claude-fable-5"


def test_build_adversary_raises_runtime_error_when_the_key_env_is_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)

    with pytest.raises(RuntimeError):
        build_adversary("fireworks-deepseek")


def test_required_key_env_returns_the_selected_adversarys_provider_key() -> None:
    # The real-mode gate keys off this, so an Anthropic adversary is gated on
    # ANTHROPIC_API_KEY, not a hardcoded FIREWORKS_API_KEY.
    assert required_key_env("fireworks-deepseek") == "FIREWORKS_API_KEY"
    assert required_key_env("anthropic-fable") == "ANTHROPIC_API_KEY"


def test_required_key_env_raises_key_error_for_an_unregistered_model_name() -> None:
    with pytest.raises(KeyError):
        required_key_env("telecom")


def test_build_adversary_defaults_to_an_empty_tool_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A dummy key is enough: client construction is lazy (no network until
    # `.complete`), so this exercises the success path offline.
    monkeypatch.setenv("FIREWORKS_API_KEY", "dummy-key")

    adversary = build_adversary("fireworks-deepseek")

    assert adversary.tool_context == ""  # type: ignore[attr-defined]


def test_build_adversary_threads_tool_context_onto_the_adversary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The grounded live run picks a registered adversary AND grounds it in
    # the agent's real tools -- build_adversary must forward tool_context.
    monkeypatch.setenv("FIREWORKS_API_KEY", "dummy-key")

    adversary = build_adversary("fireworks-deepseek", tool_context="GROUNDED-CTX")

    assert adversary.tool_context == "GROUNDED-CTX"  # type: ignore[attr-defined]
