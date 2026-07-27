from __future__ import annotations

import pytest

from bossyk_sandbox.runtime.agent_models import (
    AGENT_MODELS,
    WORKER_VLLM_HUB_ID,
    AgentModelSpec,
    agent_model,
    openai_base_url_for,
    required_key_env,
)


def test_agent_model_spec_field_defaults() -> None:
    spec = AgentModelSpec(
        name="test-model",
        provider="runpod_vllm",
        model_id="org/Model",
        hf_repo="org/Model",
        key_env="RUNPOD_API_KEY",
        static_base_url=None,
    )

    assert spec.gated is False
    assert spec.tool_call_parser is None
    assert spec.enable_auto_tool_choice is True
    assert spec.gpu_pool_id is None
    assert spec.max_model_len == 8192
    assert spec.caching is None
    assert spec.datacenter_id is None
    assert spec.idle_timeout_s == 300
    assert spec.disqualified_reason is None


def test_worker_vllm_hub_id_is_the_raw_listing_id_not_the_slug() -> None:
    # GOTCHA 1: the slug ("runpod-workers/worker-vllm") errors "failed to get
    # hub listing" -- the raw id is the only form that works.
    assert WORKER_VLLM_HUB_ID == "cm8h09d9n000008jvh2rqdsmb"
    assert "/" not in WORKER_VLLM_HUB_ID


def test_agent_model_looks_up_a_registered_spec() -> None:
    spec = agent_model("qwen2.5-7b")

    assert spec.name == "qwen2.5-7b"


def test_agent_model_raises_key_error_for_an_unregistered_name() -> None:
    with pytest.raises(KeyError):
        agent_model("no-such-model")


def test_required_key_env_returns_the_selected_models_provider_key() -> None:
    assert required_key_env("qwen2.5-7b") == "RUNPOD_API_KEY"
    assert required_key_env("kimi") == "FIREWORKS_API_KEY"


def test_required_key_env_raises_key_error_for_an_unregistered_name() -> None:
    with pytest.raises(KeyError):
        required_key_env("no-such-model")


def test_openai_base_url_for_builds_the_runpod_openai_route() -> None:
    assert (
        openai_base_url_for("wkdqe0qef23jy2") == "https://api.runpod.ai/v2/wkdqe0qef23jy2/openai/v1"
    )


# --- seed registry entries --------------------------------------------------


def test_qwen_spec_is_ungated_hermes_on_the_ada_24_pool_with_model_reference_caching() -> None:
    spec = AGENT_MODELS["qwen2.5-7b"]

    assert spec.provider == "runpod_vllm"
    assert spec.model_id == "Qwen/Qwen2.5-7B-Instruct"
    assert spec.hf_repo == "Qwen/Qwen2.5-7B-Instruct"
    assert spec.key_env == "RUNPOD_API_KEY"
    assert spec.gated is False
    assert spec.tool_call_parser == "hermes"
    assert spec.gpu_pool_id == "ADA_24"
    assert spec.caching == "model_reference"


def test_llama_spec_is_gated_llama3_json_with_network_volume_caching() -> None:
    spec = AGENT_MODELS["llama-3.1-8b"]

    assert spec.provider == "runpod_vllm"
    assert spec.model_id == "meta-llama/Llama-3.1-8B-Instruct"
    assert spec.hf_repo == "meta-llama/Llama-3.1-8B-Instruct"
    assert spec.key_env == "RUNPOD_API_KEY"
    assert spec.gated is True
    assert spec.tool_call_parser == "llama3_json"
    assert spec.caching == "network_volume"


def test_kimi_spec_is_fireworks_with_a_static_base_url_and_no_hf_repo() -> None:
    spec = AGENT_MODELS["kimi"]

    assert spec.provider == "fireworks"
    assert spec.model_id == "accounts/fireworks/models/kimi-k2p6"
    assert spec.hf_repo is None
    assert spec.key_env == "FIREWORKS_API_KEY"
    assert spec.static_base_url == "https://api.fireworks.ai/inference/v1"


def test_registry_names_match_their_spec_name_field() -> None:
    # lookup key and label must agree, mirroring adversary_registry's
    # ADVERSARY_MODELS invariant (test_adversary_registry.py has no explicit
    # equivalent, but every entry there also satisfies key == spec.name).
    for key, spec in AGENT_MODELS.items():
        assert key == spec.name
