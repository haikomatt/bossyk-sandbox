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
