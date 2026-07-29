"""Persistent GPU-pod lifecycle for the interpretability track (pure renderers).

The white-box counterpart to `runtime.runpod_serving`. That module drives a
*serverless* worker-vllm endpoint; interp needs a *persistent pod* instead,
because (a) TransformerLens/nnsight run the model directly for activation
capture -- a job, not an OpenAI endpoint -- and (b) the storage decision was
persistent-pod-with-local-NVMe, no network volume: download the (gated) Llama
weights once to the pod's disk, keep them across stop/start, and skip the
serverless-gated-volume datacenter pin entirely.

One pod serves both interp phases sequentially: `vllm serve` for the behavioral
(logprob) capture, then an nnsight job for the activation capture -- same weights
on disk, one bring-up.

Same seam as `runpod_serving`, same HARD RULE: nothing here makes a real
RunPod/HTTP call. Every renderer returns argv; the imperative wrappers hand argv
to an INJECTED `run` (production: a `subprocess.run` wrapper with
`RUNPOD_API_KEY` in its environment; tests: a fake that records argv). No secret
is ever embedded or received: a gated model's `HF_TOKEN` is rendered as the
literal shell-variable reference `$HF_TOKEN` (never a resolved key), exactly as
`runpod_serving.deploy_argv` does; the injected `run`'s environment is
responsible for resolving it on the wire.

`stop` vs `delete` is load-bearing: **stop** keeps the pod's disk (weights
survive, cheap to resume) and is the between-sessions default; **delete**
reclaims the disk and forces a re-download -- an explicit teardown, never
automatic. Every created pod also carries an auto-stop deadline
(`--terminate-after`), the enforced answer to the compute plan's "orphaned pod
overnight is the $50-killer".
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from bossyk_sandbox.runtime.agent_models import AgentModelSpec

# Local (on-pod) HuggingFace cache path. On a persistent pod this lives on the
# container disk and survives stop/start, so gated weights download once. No
# network volume (the storage decision), so this is a plain pod-disk path.
POD_HF_HOME = "/workspace/hf"


@dataclass(frozen=True)
class InterpPodSpec:
    """Infra config for the interp pod. `gpu_id` and `image` have NO defaults on
    purpose: they are real RunPod identifiers that must be confirmed at bring-up
    (`runpodctl gpu list`, `runpodctl template search`) rather than guessed --
    the same discipline that kept `WORKER_VLLM_HUB_ID` a confirmed id.

    Recommended at bring-up: an Ampere-48 class `gpu_id` (e.g. "NVIDIA A40") for
    Llama-3.1-8B in bf16 plus an nnsight activation cache with headroom, and a
    CUDA PyTorch image into which vLLM + nnsight install cleanly.
    """

    name: str
    gpu_id: str
    image: str
    container_disk_gb: int = 80  # ~16GB bf16 weights + activation caches + headroom
    http_port: int = 8000  # the vLLM OpenAI server port
    min_cuda_version: str | None = None


def _create_env(model: AgentModelSpec, *, hf_token_present: bool) -> dict[str, str]:
    """The pod's environment. `HF_TOKEN` is the literal `$HF_TOKEN` placeholder
    for a gated model (never a resolved secret); resolution is the injected
    `run`'s job. `MODEL_NAME` mirrors the registry so the on-pod serve/capture
    commands and the agent's `AGENT_MODEL` agree."""
    env: dict[str, str] = {
        "MODEL_NAME": model.model_id,
        "HF_HOME": POD_HF_HOME,
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
    }
    if model.gated:
        if not hf_token_present:
            raise ValueError(
                f"{model.name!r} is gated and needs HF_TOKEN present to pull weights "
                "(pass hf_token_present=True; the token reaches the pod via the injected run, "
                "never via argv)"
            )
        env["HF_TOKEN"] = "$HF_TOKEN"
    return env


def create_argv(
    model: AgentModelSpec,
    pod: InterpPodSpec,
    *,
    hf_token_present: bool = False,
    terminate_after: str | None = None,
) -> list[str]:
    """Render `runpodctl pod create` argv -- never executes. Exposes an HTTP port
    for vLLM and a TCP port for SSH (nnsight jobs / setup), sizes the container
    disk for the weights + activation caches, and passes the environment as the
    JSON object `--env` expects (sorted keys -> deterministic argv). Adds
    `--terminate-after` when given (the auto-stop safety)."""
    env_json = json.dumps(_create_env(model, hf_token_present=hf_token_present), sort_keys=True)
    argv = [
        "runpodctl",
        "pod",
        "create",
        "--name",
        pod.name,
        "--gpu-id",
        pod.gpu_id,
        "--image",
        pod.image,
        "--container-disk-in-gb",
        str(pod.container_disk_gb),
        "--ports",
        f"{pod.http_port}/http,22/tcp",
        "--cloud-type",
        "SECURE",
        "--env",
        env_json,
    ]
    if pod.min_cuda_version is not None:
        argv += ["--min-cuda-version", pod.min_cuda_version]
    if terminate_after is not None:
        argv += ["--terminate-after", terminate_after]
    return argv


def start_argv(pod_id: str) -> list[str]:
    """Resume a stopped pod (disk + weights intact)."""
    return ["runpodctl", "pod", "start", pod_id]


def stop_argv(pod_id: str) -> list[str]:
    """Stop a running pod, KEEPING its disk -- the between-sessions default so
    weights survive and the next session resumes cheaply."""
    return ["runpodctl", "pod", "stop", pod_id]


def terminate_argv(pod_id: str) -> list[str]:
    """Delete a pod, reclaiming its disk (weights are lost, re-download on
    recreate). Explicit teardown -- never automatic."""
    return ["runpodctl", "pod", "delete", pod_id]


def status_argv() -> list[str]:
    """List all pods -- the `./pod status` run at the start of every session to
    catch an orphaned pod before it bills overnight."""
    return ["runpodctl", "pod", "list"]


def vllm_serve_command(model: AgentModelSpec, pod: InterpPodSpec) -> list[str]:
    """The command run ON the pod to bring up the OpenAI-compatible vLLM server
    for the behavioral phase (the `./pod run` step). Tool-calling flags mirror
    the model's registry entry so the agent binds tools identically to the
    serverless path; vLLM returns per-token logprobs on its OpenAI route without
    a special flag, so behavioral capture needs no extra server config."""
    argv = [
        "vllm",
        "serve",
        model.model_id,
        "--port",
        str(pod.http_port),
        "--max-model-len",
        str(model.max_model_len),
    ]
    if model.enable_auto_tool_choice:
        argv.append("--enable-auto-tool-choice")
    if model.tool_call_parser is not None:
        argv += ["--tool-call-parser", model.tool_call_parser]
    return argv


def create(
    model: AgentModelSpec,
    pod: InterpPodSpec,
    *,
    hf_token_present: bool = False,
    terminate_after: str,
    run: Callable[[list[str]], Any],
) -> Any:
    """Create the pod via the injected `run`. `terminate_after` is REQUIRED here
    (not on the pure renderer) so the billable seam cannot create a pod without
    an auto-stop deadline -- the hard-cap-in-the-runnable-script rule, enforced by
    construction. `run` is the only thing that may reach a real RunPod call."""
    if not terminate_after:
        raise ValueError(
            "create() requires a terminate_after deadline (auto-stop safety); "
            "compute it in the caller and pass it, e.g. now + a few hours in ISO-8601"
        )
    argv = create_argv(
        model, pod, hf_token_present=hf_token_present, terminate_after=terminate_after
    )
    return run(argv)


def start(pod_id: str, *, run: Callable[[list[str]], Any]) -> Any:
    return run(start_argv(pod_id))


def stop(pod_id: str, *, run: Callable[[list[str]], Any]) -> Any:
    return run(stop_argv(pod_id))


def terminate(pod_id: str, *, run: Callable[[list[str]], Any]) -> Any:
    return run(terminate_argv(pod_id))


def status(*, run: Callable[[list[str]], Any]) -> Any:
    return run(status_argv())
