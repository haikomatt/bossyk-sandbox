from __future__ import annotations

import os
from dataclasses import dataclass

from bossyk_sandbox.conditions.adversary import Adversary
from bossyk_sandbox.conditions.adversary_providers import (
    build_anthropic_client,
    build_openai_compatible_client,
)
from bossyk_sandbox.conditions.fireworks_adversary import ChatClient, FireworksAdversary


@dataclass(frozen=True)
class AdversaryModelSpec:
    """One entry in the multi-provider adversary registry -- maps a model
    name to the provider client it needs, mirroring the
    `BOUNDARIES_BY_DOMAIN` registry-lookup pattern (conditions/grid.py)."""

    name: str
    provider: str  # "openai_compatible" | "anthropic"
    base_url: str | None
    key_env: str
    model_id: str


ADVERSARY_MODELS: dict[str, AdversaryModelSpec] = {
    "fireworks-deepseek": AdversaryModelSpec(
        name="fireworks-deepseek",
        provider="openai_compatible",
        base_url="https://api.fireworks.ai/inference/v1",
        key_env="FIREWORKS_API_KEY",
        model_id="accounts/fireworks/models/deepseek-v4-pro",
    ),
    "anthropic-fable": AdversaryModelSpec(
        name="anthropic-fable",
        provider="anthropic",
        base_url=None,
        key_env="ANTHROPIC_API_KEY",
        model_id="claude-fable-5",
    ),
}


def required_key_env(name: str) -> str:
    """The environment variable that must hold `name`'s provider API key to run
    it in real mode -- the real-mode gate keys off this, so each adversary is
    gated on its own provider's key (not a hardcoded one). Raises `KeyError`
    for an unregistered name, matching `build_adversary`."""
    return ADVERSARY_MODELS[name].key_env


def build_adversary(
    name: str, *, tool_context: str = "", fallback_model: str | None = None
) -> Adversary:
    """Looks up `name` in `ADVERSARY_MODELS` and builds the matching real
    `Adversary` -- the client-agnostic `FireworksAdversary` wrapping
    whichever provider client the spec calls for. Raises `KeyError` for an
    unregistered name, `RuntimeError` if the model's API key env var is
    unset. `fallback_model` only applies to the anthropic provider (passed
    through to `build_anthropic_client`). `tool_context` grounds the built
    adversary in the target agent's real tools (see
    conditions.fireworks_adversary.render_tool_context)."""
    spec = ADVERSARY_MODELS[name]
    api_key = os.environ.get(spec.key_env)
    if not api_key:
        raise RuntimeError(
            f"{spec.key_env} is required to run the {name!r} adversary "
            f"(set it in .env, matching build_fireworks_adversary's convention)."
        )
    client: ChatClient
    if spec.provider == "openai_compatible":
        if spec.base_url is None:
            raise ValueError(f"adversary spec {name!r} is openai_compatible but has no base_url")
        client = build_openai_compatible_client(
            base_url=spec.base_url, api_key=api_key, model=spec.model_id
        )
    else:
        client = build_anthropic_client(model=spec.model_id, fallback_model=fallback_model)
    return FireworksAdversary(client=client, model=spec.model_id, tool_context=tool_context)
