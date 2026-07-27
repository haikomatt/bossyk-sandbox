"""RunPod lifecycle helper: pure argv/payload renderers (encode the 5 deploy
gotchas, see session-handoff-2026-07-27-runpod-serving) plus thin
deploy()/warm()/teardown() behind an injected run/post. HARD RULE: nothing
here may make a real RunPod/HTTP call -- every test injects a fake `run`/
`post` and asserts on the rendered argv/payload/URL, never a live response.
"""

from __future__ import annotations

from typing import Any

import pytest

from bossyk_sandbox.runtime.agent_models import AGENT_MODELS, WORKER_VLLM_HUB_ID, AgentModelSpec
from bossyk_sandbox.runtime.runpod_serving import (
    deploy,
    deploy_argv,
    idle_timeout_argv,
    openai_base_url_for,
    teardown,
    teardown_argv,
    warm,
    warm_poll_target,
    warm_request_payload,
)

QWEN = AGENT_MODELS["qwen2.5-7b"]
LLAMA = AGENT_MODELS["llama-3.1-8b"]
KIMI = AGENT_MODELS["kimi"]


# --- deploy_argv (GOTCHA 1, 2, 5) --------------------------------------------


def test_deploy_argv_ungated_uses_model_reference_and_the_raw_hub_id() -> None:
    argv = deploy_argv(QWEN)

    assert argv[:3] == ["runpodctl", "serverless", "create"]
    assert argv[argv.index("--hub-id") + 1] == WORKER_VLLM_HUB_ID  # GOTCHA 1: raw id, not the slug
    assert argv[argv.index("--gpu-id") + 1] == "ADA_24"
    assert "MODEL_NAME=Qwen/Qwen2.5-7B-Instruct" in argv
    assert "TOOL_CALL_PARSER=hermes" in argv
    assert argv[argv.index("--model-reference") + 1] == (
        "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct:main"
    )
    assert "HF_TOKEN=$HF_TOKEN" not in argv  # ungated: no token needed
    assert "--network-volume-id" not in argv
    assert argv[-4:] == ["--workers-min", "0", "--workers-max", "1"]


def test_deploy_argv_gated_drops_model_reference_and_adds_hf_token_and_volume() -> None:
    # GOTCHA 2: RunPod validates --model-reference server-side WITHOUT the
    # token, so gated models must drop it and let the WORKER download with
    # HF_TOKEN instead. GOTCHA 5: the download must land on a network volume
    # or an uncached gated cold-start hangs (a stuck Llama burned ~30 min).
    argv = deploy_argv(LLAMA, network_volume_id="vol-123", hf_token_present=True)

    assert "--model-reference" not in argv
    assert "HF_TOKEN=$HF_TOKEN" in argv
    assert "HF_HUB_CACHE=/runpod-volume/hf-cache" in argv
    assert argv[argv.index("--network-volume-id") + 1] == "vol-123"
    assert "TOOL_CALL_PARSER=llama3_json" in argv


def test_deploy_argv_gated_without_hf_token_present_raises() -> None:
    with pytest.raises(ValueError, match="HF_TOKEN"):
        deploy_argv(LLAMA, network_volume_id="vol-123", hf_token_present=False)


def test_deploy_argv_gated_without_network_volume_id_raises() -> None:
    # GOTCHA 5 mitigation: a network volume is mandatory for gated models,
    # never inferred/defaulted.
    with pytest.raises(ValueError, match="network volume"):
        deploy_argv(LLAMA, network_volume_id=None, hf_token_present=True)


def test_deploy_argv_omits_tool_call_parser_env_when_undiscovered() -> None:
    # None = undiscovered (e.g. Gemma before it's smoke-tested) -- must not
    # emit a bogus TOOL_CALL_PARSER env.
    undiscovered = AgentModelSpec(
        name="mystery-model",
        provider="runpod_vllm",
        model_id="org/Mystery",
        hf_repo="org/Mystery",
        key_env="RUNPOD_API_KEY",
        static_base_url=None,
        tool_call_parser=None,
        gpu_pool_id="ADA_24",
    )

    argv = deploy_argv(undiscovered)

    assert not any(item.startswith("TOOL_CALL_PARSER=") for item in argv)


def test_deploy_argv_rejects_a_non_runpod_provider_spec() -> None:
    with pytest.raises(ValueError, match="runpod_vllm"):
        deploy_argv(KIMI)


def test_deploy_argv_rejects_a_spec_with_no_gpu_pool_id() -> None:
    no_pool = AgentModelSpec(
        name="no-pool",
        provider="runpod_vllm",
        model_id="org/Model",
        hf_repo="org/Model",
        key_env="RUNPOD_API_KEY",
        static_base_url=None,
    )

    with pytest.raises(ValueError, match="gpu_pool_id"):
        deploy_argv(no_pool)


# --- idle_timeout_argv (GOTCHA 4) --------------------------------------------


def test_idle_timeout_argv_renders_the_serverless_update_command() -> None:
    argv = idle_timeout_argv("wkdqe0qef23jy2", 300)

    assert argv == ["runpodctl", "serverless", "update", "wkdqe0qef23jy2", "--idle-timeout", "300"]


# --- warm renderers (GOTCHA 3: async /run + poll /status, never sync /openai)


def test_warm_request_payload_wraps_the_openai_route_for_the_job_api() -> None:
    payload = warm_request_payload(QWEN)

    assert payload["input"]["openai_route"] == "/v1/chat/completions"
    assert payload["input"]["openai_input"]["model"] == QWEN.model_id
    assert payload["input"]["openai_input"]["messages"]


def test_warm_poll_target_is_completed() -> None:
    assert warm_poll_target() == "COMPLETED"


def test_teardown_argv_renders_the_serverless_delete_command() -> None:
    assert teardown_argv("wkdqe0qef23jy2") == ["runpodctl", "serverless", "delete", "wkdqe0qef23jy2"]


def test_openai_base_url_for_is_shared_with_the_registry_not_redefined() -> None:
    from bossyk_sandbox.runtime import agent_models

    assert openai_base_url_for is agent_models.openai_base_url_for


# --- thin deploy()/warm()/teardown() behind an injected run/post -----------


def test_deploy_hands_the_rendered_argv_to_the_injected_run() -> None:
    seen: list[list[str]] = []

    def fake_run(argv: list[str]) -> dict[str, str]:
        seen.append(argv)
        return {"id": "new-endpoint-id"}

    result = deploy(QWEN, run=fake_run)

    assert seen == [deploy_argv(QWEN)]
    assert result == {"id": "new-endpoint-id"}


def test_teardown_hands_the_rendered_argv_to_the_injected_run() -> None:
    seen: list[list[str]] = []

    def fake_run(argv: list[str]) -> dict[str, str]:
        seen.append(argv)
        return {"ok": True}

    result = teardown("wkdqe0qef23jy2", run=fake_run)

    assert seen == [teardown_argv("wkdqe0qef23jy2")]
    assert result == {"ok": True}


def test_warm_posts_the_run_job_then_polls_status_until_completed() -> None:
    responses = [
        {"id": "job-1", "status": "IN_QUEUE"},
        {"status": "IN_PROGRESS"},
        {"status": "COMPLETED"},
    ]
    calls: list[tuple[str, dict[str, Any] | None]] = []

    def fake_post(url: str, payload: dict[str, Any] | None) -> dict[str, Any]:
        calls.append((url, payload))
        return responses[len(calls) - 1]

    slept: list[float] = []

    result = warm(QWEN, "ep-1", post=fake_post, sleeper=slept.append)

    assert result == {"job_id": "job-1", "status": "COMPLETED"}
    assert calls[0] == ("https://api.runpod.ai/v2/ep-1/run", warm_request_payload(QWEN))
    assert calls[1] == ("https://api.runpod.ai/v2/ep-1/status/job-1", None)
    assert calls[2] == ("https://api.runpod.ai/v2/ep-1/status/job-1", None)
    assert len(slept) == 2  # bounded, never a sync /openai call


def test_warm_never_hits_a_sync_openai_route_on_a_cold_worker() -> None:
    # GOTCHA 3: sync /openai 524s on a cold worker. `warm` must only ever
    # hit /run and /status.
    calls: list[str] = []

    def fake_post(url: str, _payload: dict[str, Any] | None) -> dict[str, Any]:
        calls.append(url)
        return {"id": "job-1", "status": "COMPLETED"}

    warm(QWEN, "ep-1", post=fake_post, sleeper=lambda _s: None)

    assert all("/openai" not in url for url in calls)


def test_warm_raises_instead_of_hanging_when_never_completed() -> None:
    # fail-not-hang: bounded polling, never an unbounded wait (mirrors
    # live_h2h4_bench.replay_with_retry's bounded-retry discipline).
    def fake_post(url: str, _payload: dict[str, Any] | None) -> dict[str, Any]:
        return {"id": "job-1", "status": "IN_PROGRESS"}

    with pytest.raises(TimeoutError):
        warm(QWEN, "ep-1", post=fake_post, max_polls=3, sleeper=lambda _s: None)
