"""Hermetic tests for runtime.interp_pod: pure argv renderers + injected run.

HARD RULE (inherited from test_runpod_serving): nothing here makes a real
RunPod/HTTP call, and no real secret ever appears -- a gated model's token is
only ever the literal `$HF_TOKEN` placeholder.
"""

from __future__ import annotations

import json

import pytest

from bossyk_sandbox.runtime.agent_models import agent_model
from bossyk_sandbox.runtime.interp_pod import (
    InterpPodSpec,
    create,
    create_argv,
    start_argv,
    status_argv,
    stop_argv,
    terminate_argv,
    vllm_serve_command,
)

QWEN = agent_model("qwen2.5-7b")  # ungated
LLAMA = agent_model("llama-3.1-8b")  # gated

POD = InterpPodSpec(name="interp-pod", gpu_id="NVIDIA A40", image="runpod/pytorch:cuda")


def _flag(argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


def _env(argv: list[str]) -> dict[str, str]:
    return json.loads(_flag(argv, "--env"))


def test_create_argv_shape_and_ports() -> None:
    argv = create_argv(QWEN, POD)
    assert argv[:3] == ["runpodctl", "pod", "create"]
    assert _flag(argv, "--name") == "interp-pod"
    assert _flag(argv, "--gpu-id") == "NVIDIA A40"
    assert _flag(argv, "--image") == "runpod/pytorch:cuda"
    assert _flag(argv, "--container-disk-in-gb") == "80"
    assert _flag(argv, "--ports") == "8000/http,22/tcp"
    assert _flag(argv, "--cloud-type") == "SECURE"


def test_create_argv_ungated_has_no_hf_token() -> None:
    env = _env(create_argv(QWEN, POD))
    assert "HF_TOKEN" not in env
    assert env["MODEL_NAME"] == "Qwen/Qwen2.5-7B-Instruct"
    assert env["HF_HOME"] == "/workspace/hf"
    assert env["HF_HUB_ENABLE_HF_TRANSFER"] == "1"


def test_create_argv_gated_uses_literal_token_placeholder_never_a_secret() -> None:
    argv = create_argv(LLAMA, POD, hf_token_present=True)
    # GATED: the token is the literal shell-variable reference, never resolved.
    assert _env(argv)["HF_TOKEN"] == "$HF_TOKEN"
    # belt-and-braces: no argv element is a real HuggingFace token (they start "hf_").
    assert not any(part.startswith("hf_") for part in argv)


def test_create_argv_gated_without_token_raises() -> None:
    with pytest.raises(ValueError, match="gated"):
        create_argv(LLAMA, POD, hf_token_present=False)


def test_create_argv_terminate_after_is_optional_and_rendered_when_given() -> None:
    assert "--terminate-after" not in create_argv(QWEN, POD)
    argv = create_argv(QWEN, POD, terminate_after="2026-07-29T00:00:00Z")
    assert _flag(argv, "--terminate-after") == "2026-07-29T00:00:00Z"


def test_create_argv_min_cuda_rendered_when_set() -> None:
    pod = InterpPodSpec(name="p", gpu_id="NVIDIA A40", image="img", min_cuda_version="12.6")
    assert _flag(create_argv(QWEN, pod), "--min-cuda-version") == "12.6"


def test_lifecycle_argv_stop_keeps_disk_terminate_deletes() -> None:
    assert stop_argv("pod-1") == ["runpodctl", "pod", "stop", "pod-1"]
    assert start_argv("pod-1") == ["runpodctl", "pod", "start", "pod-1"]
    assert terminate_argv("pod-1") == ["runpodctl", "pod", "delete", "pod-1"]
    assert status_argv() == ["runpodctl", "pod", "list"]


def test_vllm_serve_command_mirrors_registry_toolcalling() -> None:
    argv = vllm_serve_command(LLAMA, POD)
    assert argv[:3] == ["vllm", "serve", "meta-llama/Llama-3.1-8B-Instruct"]
    assert _flag(argv, "--port") == "8000"
    assert _flag(argv, "--max-model-len") == str(LLAMA.max_model_len)
    assert "--enable-auto-tool-choice" in argv
    assert _flag(argv, "--tool-call-parser") == "llama3_json"


def test_create_requires_terminate_after_deadline() -> None:
    calls: list[list[str]] = []
    with pytest.raises(ValueError, match="terminate_after"):
        create(LLAMA, POD, hf_token_present=True, terminate_after="", run=calls.append)
    assert calls == []  # nothing reached the injected run


def test_create_hands_rendered_argv_to_injected_run() -> None:
    calls: list[list[str]] = []
    create(
        LLAMA,
        POD,
        hf_token_present=True,
        terminate_after="2026-07-29T00:00:00Z",
        run=calls.append,
    )
    assert calls == [
        create_argv(LLAMA, POD, hf_token_present=True, terminate_after="2026-07-29T00:00:00Z")
    ]


def test_lifecycle_wrappers_never_touch_network() -> None:
    # A run that explodes if called proves the renderers are pure; the wrappers
    # only call run with argv, never a real client.
    def boom(_argv: list[str]) -> None:
        raise AssertionError("no real RunPod call allowed in tests")

    # pure renderers: safe to call, no run
    create_argv(LLAMA, POD, hf_token_present=True)
    stop_argv("x")
    status_argv()
    # the wrapper below WOULD call run -> boom; we assert it does exactly that
    with pytest.raises(AssertionError, match="no real RunPod call"):
        create(QWEN, POD, terminate_after="2026-07-29T00:00:00Z", run=boom)
