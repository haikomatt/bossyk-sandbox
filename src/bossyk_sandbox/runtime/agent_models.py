"""The agent-under-test registry: mirrors
`conditions.adversary_registry`'s altitude exactly -- a frozen dataclass, a
module-level dict, and plain resolver functions. No connector class, no
ModelFactory: the extension point for a new agent-under-test is "add a spec
entry", not "add a class" (see
docs/phase-agent-model-factory.md's scoping note).

A live RunPod serverless endpoint id is RUNTIME STATE, not a registry field
-- it is created/destroyed per session (`runtime.runpod_serving`) and
threaded through via an `endpoints: Mapping[str, str]` map at the call site
(`scripts/voice_model_sweep.py`), never stored here.
"""

from __future__ import annotations

from dataclasses import dataclass

# GOTCHA 1: this MUST be the raw hub listing id, never the
# "runpod-workers/worker-vllm" slug -- the slug errors "failed to get hub
# listing" on `runpodctl serverless create --hub-id`. Retrieved via
# `runpodctl hub get runpod-workers/worker-vllm`.
WORKER_VLLM_HUB_ID = "cm8h09d9n000008jvh2rqdsmb"


@dataclass(frozen=True)
class AgentModelSpec:
    """One entry in the multi-provider agent-under-test registry -- maps a
    model name to the provider it runs on plus every worker-vllm deploy
    knob that provider needs, so adding a model is one dict entry instead
    of re-deriving RunPod gotchas from prose each time."""

    name: str
    provider: str  # "runpod_vllm" | "fireworks"
    model_id: str  # OpenAI `model` / AGENT_MODEL / worker-vllm MODEL_NAME
    hf_repo: str | None  # HF repo for worker-vllm; None for Fireworks
    key_env: str  # env var holding this model's inference key
    static_base_url: str | None  # Fireworks fixed URL; None for RunPod (derived per endpoint)
    gated: bool = False  # RunPod: drop --model-reference, need HF_TOKEN + network volume
    tool_call_parser: str | None = None  # "hermes" | "llama3_json" | ...; None = undiscovered
    enable_auto_tool_choice: bool = True  # worker-vllm ENABLE_AUTO_TOOL_CHOICE
    gpu_pool_id: str | None = None  # RunPod pool id (ADA_24, AMPERE_48, ADA_80), not a display name
    max_model_len: int = 8192  # worker-vllm MAX_MODEL_LEN
    caching: str | None = None  # "model_reference" (ungated host-cache) | "network_volume" (gated)
    datacenter_id: str | None = None  # region pin for the network volume (volume is region-scoped)
    idle_timeout_s: int = 300  # keep the worker warm across a run (GOTCHA 4)
    disqualified_reason: str | None = None  # first-class finding slot; None = eligible


AGENT_MODELS: dict[str, AgentModelSpec] = {
    "qwen2.5-7b": AgentModelSpec(
        name="qwen2.5-7b",
        provider="runpod_vllm",
        model_id="Qwen/Qwen2.5-7B-Instruct",
        hf_repo="Qwen/Qwen2.5-7B-Instruct",
        key_env="RUNPOD_API_KEY",
        static_base_url=None,
        gated=False,
        tool_call_parser="hermes",
        gpu_pool_id="ADA_24",
        caching="model_reference",
    ),
    "llama-3.1-8b": AgentModelSpec(
        name="llama-3.1-8b",
        provider="runpod_vllm",
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        hf_repo="meta-llama/Llama-3.1-8B-Instruct",
        key_env="RUNPOD_API_KEY",
        static_base_url=None,
        gated=True,
        tool_call_parser="llama3_json",
        gpu_pool_id="ADA_24",
        caching="network_volume",
    ),
    "kimi": AgentModelSpec(
        name="kimi",
        provider="fireworks",
        model_id="accounts/fireworks/models/kimi-k2p6",
        hf_repo=None,
        key_env="FIREWORKS_API_KEY",
        static_base_url="https://api.fireworks.ai/inference/v1",
    ),
}


def agent_model(name: str) -> AgentModelSpec:
    """Look up `name` in `AGENT_MODELS`. Raises `KeyError` for an
    unregistered name, matching `adversary_registry.build_adversary`."""
    return AGENT_MODELS[name]


def required_key_env(name: str) -> str:
    """The environment variable that must hold `name`'s provider API key --
    each agent-under-test is gated on its own provider's key, not a
    hardcoded one. Raises `KeyError` for an unregistered name."""
    return AGENT_MODELS[name].key_env


def openai_base_url_for(endpoint_id: str) -> str:
    """RunPod's OpenAI-compatible base URL for a live serverless endpoint id
    -- pure string formatting, shared by the registry (AGENT_BASE_URL
    resolution in `scripts/voice_model_sweep.py`) and
    `runtime.runpod_serving`'s deploy/warm helpers, so the URL shape is
    defined exactly once."""
    return f"https://api.runpod.ai/v2/{endpoint_id}/openai/v1"
