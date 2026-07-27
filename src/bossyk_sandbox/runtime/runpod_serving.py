"""RunPod serverless worker-vllm lifecycle helper: pure argv/payload
renderers that encode the 5 deploy gotchas from
session-handoff-2026-07-27-runpod-serving (raw hub id, drop
--model-reference for gated models, async-warm instead of a sync cold call,
raised idle-timeout, network-volume caching for gated models), plus a thin
imperative layer (`deploy`/`warm`/`teardown`) behind an INJECTED `run`/
`post` -- the same seam shape as `scripts/voice_model_sweep.py::run_sweep`'s
injected `run_bench`.

HARD RULE: this module must never make a real RunPod/HTTP call from the test
suite. `run`/`post` are always caller-supplied (`subprocess.run`/an httpx
wrapper in production; fakes in every test) -- nothing here reaches for
`subprocess` or `httpx` directly, and nothing here ever receives or embeds a
real API key: gated deploys reference `$HF_TOKEN` as a literal shell
variable name (never a resolved secret), and `RUNPOD_API_KEY` is expected to
reach `runpodctl` via the injected `run`'s own environment, never via argv.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from bossyk_sandbox.runtime import agent_models
from bossyk_sandbox.runtime.agent_models import AgentModelSpec

# Re-exported so callers needing "the RunPod URL builder" can import it from
# either module; the definition lives once in the registry (agent_models.py)
# -- runpod_serving reuses it rather than keeping its own copy of the URL
# shape.
openai_base_url_for = agent_models.openai_base_url_for


def deploy_argv(
    spec: AgentModelSpec,
    *,
    network_volume_id: str | None = None,
    hf_token_present: bool = False,
) -> list[str]:
    """Render the `runpodctl serverless create` argv for `spec` -- never
    executes anything. GOTCHA 1: always the raw `WORKER_VLLM_HUB_ID`, never
    the hub slug. GOTCHA 2 + 5: a gated model drops `--model-reference`
    (RunPod validates it server-side WITHOUT the HF token) and instead sets
    `HF_TOKEN` + `HF_HUB_CACHE=/runpod-volume/hf-cache` + a mandatory
    network volume, so the worker downloads once and caches instead of
    re-pulling the full gated weights on every cold start. An ungated model
    uses `--model-reference` for RunPod's host-side cache instead.

    `hf_token_present` is a boolean flag, not the token value: the rendered
    `HF_TOKEN=$HF_TOKEN` is a literal shell-variable reference (mirroring
    the handoff's manual recipe), never a resolved secret -- this function
    must never see, let alone embed, a real key."""
    if spec.provider != "runpod_vllm":
        raise ValueError(
            f"{spec.name!r} is provider={spec.provider!r}, not runpod_vllm -- nothing to deploy"
        )
    if not spec.gpu_pool_id:
        raise ValueError(f"{spec.name!r} has no gpu_pool_id set -- required for --gpu-id")

    argv = [
        "runpodctl",
        "serverless",
        "create",
        "--name",
        spec.name,
        "--hub-id",
        agent_models.WORKER_VLLM_HUB_ID,
        "--gpu-id",
        spec.gpu_pool_id,
        "--env",
        f"MODEL_NAME={spec.model_id}",
        "--env",
        f"ENABLE_AUTO_TOOL_CHOICE={'true' if spec.enable_auto_tool_choice else 'false'}",
    ]
    if spec.tool_call_parser is not None:
        argv += ["--env", f"TOOL_CALL_PARSER={spec.tool_call_parser}"]
    argv += ["--env", f"MAX_MODEL_LEN={spec.max_model_len}"]

    if spec.gated:
        if not hf_token_present:
            raise ValueError(
                f"{spec.name!r} is gated and needs HF_TOKEN present to deploy (GOTCHA 2)"
            )
        if not network_volume_id:
            raise ValueError(
                f"{spec.name!r} is gated and needs a network volume id -- an uncached gated "
                "cold-start hangs (GOTCHA 5)"
            )
        argv += [
            "--env",
            "HF_TOKEN=$HF_TOKEN",
            "--env",
            "HF_HUB_CACHE=/runpod-volume/hf-cache",
            "--network-volume-id",
            network_volume_id,
        ]
    else:
        if not spec.hf_repo:
            raise ValueError(f"{spec.name!r} is ungated but has no hf_repo for --model-reference")
        argv += ["--model-reference", f"https://huggingface.co/{spec.hf_repo}:main"]

    argv += ["--workers-min", "0", "--workers-max", "1"]
    return argv


def idle_timeout_argv(endpoint_id: str, idle_timeout_s: int) -> list[str]:
    """GOTCHA 4: raise the endpoint's idle-timeout (default ~10s) so the
    worker survives the gap between sweep attempts instead of cold-starting
    on every attack."""
    return [
        "runpodctl",
        "serverless",
        "update",
        endpoint_id,
        "--idle-timeout",
        str(idle_timeout_s),
    ]


def warm_request_payload(spec: AgentModelSpec) -> dict[str, Any]:
    """The async `/run` job body that starts a worker on a scale-to-zero
    endpoint. GOTCHA 3: a sync `/openai` call 524s on a cold worker (~100s
    Cloudflare edge timeout) -- the job API is the only safe way to trigger
    a cold start. worker-vllm's OpenAI route is wrapped in
    `openai_route`/`openai_input` for the job API (distinct from the
    persistent `/openai/v1` proxy `AGENT_BASE_URL` uses once warm).
    `max_tokens=1` keeps the warm-up call cheap -- it exists only to start a
    worker, not to produce a real completion."""
    return {
        "input": {
            "openai_route": "/v1/chat/completions",
            "openai_input": {
                "model": spec.model_id,
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 1,
            },
        }
    }


def warm_poll_target() -> str:
    """The job status `warm()` polls `/status/<job-id>` for (GOTCHA 3):
    warming goes through the async job API and polls until the job reaches
    this terminal status, rather than ever making a sync call that could
    524 on a cold worker."""
    return "COMPLETED"


def teardown_argv(endpoint_id: str) -> list[str]:
    """GOTCHA-adjacent housekeeping: delete a scale-to-zero endpoint when a
    session is done (teardown stays an explicit step, never automatic --
    see the plan's scope boundaries)."""
    return ["runpodctl", "serverless", "delete", endpoint_id]


def deploy(
    spec: AgentModelSpec,
    *,
    network_volume_id: str | None = None,
    hf_token_present: bool = False,
    run: Callable[[list[str]], Any],
) -> Any:
    """Render `deploy_argv` and hand it to the injected `run` (production:
    a `subprocess.run` wrapper with `RUNPOD_API_KEY` in its environment;
    tests: a fake that records the argv). `run` is the ONLY thing that may
    ever reach a real RunPod call -- tests must always inject a fake."""
    argv = deploy_argv(spec, network_volume_id=network_volume_id, hf_token_present=hf_token_present)
    return run(argv)


def teardown(endpoint_id: str, *, run: Callable[[list[str]], Any]) -> Any:
    """Render `teardown_argv` and hand it to the injected `run`."""
    return run(teardown_argv(endpoint_id))


def warm(
    spec: AgentModelSpec,
    endpoint_id: str,
    *,
    post: Callable[[str, dict[str, Any] | None], dict[str, Any]],
    max_polls: int = 60,
    poll_interval_s: float = 2.0,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Warm a scale-to-zero endpoint via the async job API (GOTCHA 3): POST
    `warm_request_payload` to `/run`, then poll `/status/<job-id>` (same
    injected `post`, called with `payload=None` for a GET-shaped call) until
    `warm_poll_target()` or `max_polls` is exhausted. Bounded, fail-not-hang
    polling -- mirrors `live_h2h4_bench.replay_with_retry`'s bounded-retry
    discipline rather than an unbounded wait. Raises `TimeoutError` if the
    endpoint never reaches the target status; never falls back to a sync
    `/openai` call."""
    base_url = openai_base_url_for(endpoint_id).removesuffix("/openai/v1")
    run_response = post(f"{base_url}/run", warm_request_payload(spec))
    job_id = run_response["id"]
    status = run_response.get("status")
    target = warm_poll_target()
    polls = 0
    while status != target and polls < max_polls:
        sleeper(poll_interval_s)
        status_response = post(f"{base_url}/status/{job_id}", None)
        status = status_response.get("status")
        polls += 1
    if status != target:
        raise TimeoutError(
            f"endpoint {endpoint_id} did not reach {target!r} after {max_polls} polls "
            f"(last status: {status!r})"
        )
    return {"job_id": job_id, "status": status}
