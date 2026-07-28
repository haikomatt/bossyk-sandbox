#!/usr/bin/env python
"""Interp pod lifecycle CLI: the deterministic `./pod up/status/stop/down`
wrapper the compute plan calls for -- Claude Code authors it, invokes it, and
the script (not the model) owns the billable lifecycle.

Thin layer over `runtime.interp_pod`'s pure renderers: this module supplies the
one thing the renderers refuse to embed -- a real `run` (a `subprocess.run`
wrapper carrying `RUNPOD_API_KEY`) -- plus the auto-stop deadline and the
env-gate. Everything billable is deliberately awkward to trigger:

- `up` is gated on RUN_INTERP_POD=1 AND (for a gated model) HF_TOKEN present, and
  ALWAYS passes a `--terminate-after` deadline (default +6h) so a forgotten pod
  self-stops -- the enforced answer to "an orphaned pod overnight is the
  $50-killer".
- `stop` keeps the disk (weights survive; cheap resume); `down` deletes the pod
  (explicit teardown).
- `status` (`pod list`) is the read-only session-start orphan check.
- `serve-cmd` only PRINTS the on-pod `vllm serve ...` line -- no API call.

Side-effect-free at import (tests import this module and exercise `bring_up` /
`deadline_iso` with an injected `run` and a fixed `now`; they never call
`main()` and never touch the network).

Usage:
    uv run python scripts/interp_pod.py status
    RUN_INTERP_POD=1 HF_TOKEN=... RUNPOD_API_KEY=... \\
        uv run python scripts/interp_pod.py up \\
        --gpu-id "NVIDIA A40" --image runpod/pytorch:<cuda-tag> --hours 6
    uv run python scripts/interp_pod.py serve-cmd --model llama-3.1-8b
    uv run python scripts/interp_pod.py stop <pod_id>   # keeps disk
    uv run python scripts/interp_pod.py down <pod_id>   # deletes (teardown)

Confirm --gpu-id and --image against `runpodctl gpu list` / `runpodctl template
search` before the first `up` -- they are real ids, deliberately not defaulted.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
from typing import Any

from bossyk_sandbox.env import load_project_env
from bossyk_sandbox.runtime import interp_pod
from bossyk_sandbox.runtime.agent_models import agent_model
from bossyk_sandbox.runtime.interp_pod import InterpPodSpec

DEFAULT_MODEL = "llama-3.1-8b"
DEFAULT_HOURS = 6.0


def deadline_iso(now: datetime, hours: float) -> str:
    """RunPod `--terminate-after` timestamp: `now + hours`, ISO-8601 Z. Pure and
    `now`-injected so the auto-stop deadline is testable without a clock."""
    return (now + timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%SZ")


def _subprocess_run(argv: list[str]) -> str:
    """The real injected `run`: shell out to runpodctl with RUNPOD_API_KEY in the
    inherited environment (never in argv). Raises before spawning if the key is
    absent, so a misconfigured shell fails fast instead of erroring inside
    runpodctl."""
    if not os.environ.get("RUNPOD_API_KEY"):
        raise RuntimeError("RUNPOD_API_KEY not set in the environment")
    proc = subprocess.run(argv, check=True, capture_output=True, text=True)
    return proc.stdout


def bring_up(
    *,
    model_name: str,
    gpu_id: str,
    image: str,
    container_disk_gb: int,
    hours: float,
    now: datetime,
    env: Mapping[str, str],
    run: Callable[[list[str]], Any],
) -> tuple[Any, str]:
    """Create the interp pod. Preconditions checked here (not the RUN_INTERP_POD
    env-gate, which is main()'s job): a gated model requires HF_TOKEN in `env`.
    Computes the auto-stop deadline from the injected `now` and delegates to
    `interp_pod.create` (which itself refuses a missing deadline). Returns
    (run result, deadline) so the caller can report the teardown time."""
    model = agent_model(model_name)
    hf_token_present = bool(env.get("HF_TOKEN"))
    if model.gated and not hf_token_present:
        raise RuntimeError(
            f"{model_name!r} is gated -- set HF_TOKEN in the environment before bring-up"
        )
    pod = InterpPodSpec(
        name=f"interp-{model_name}",
        gpu_id=gpu_id,
        image=image,
        container_disk_gb=container_disk_gb,
    )
    terminate_after = deadline_iso(now, hours)
    result = interp_pod.create(
        model,
        pod,
        hf_token_present=hf_token_present,
        terminate_after=terminate_after,
        run=run,
    )
    return result, terminate_after


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="interp pod lifecycle")
    sub = parser.add_subparsers(dest="cmd", required=True)

    up = sub.add_parser("up", help="create the (billable) interp pod")
    up.add_argument("--model", default=DEFAULT_MODEL)
    up.add_argument("--gpu-id", required=True, help="from `runpodctl gpu list`")
    up.add_argument("--image", required=True, help="CUDA image with vllm+nnsight installable")
    up.add_argument("--container-disk-gb", type=int, default=80)
    up.add_argument("--hours", type=float, default=DEFAULT_HOURS, help="auto-stop after N hours")

    sub.add_parser("status", help="list pods (session-start orphan check)")

    stop_p = sub.add_parser("stop", help="stop a pod (keeps disk)")
    stop_p.add_argument("pod_id")

    down_p = sub.add_parser("down", help="delete a pod (teardown)")
    down_p.add_argument("pod_id")

    serve = sub.add_parser("serve-cmd", help="print the on-pod `vllm serve` command")
    serve.add_argument("--model", default=DEFAULT_MODEL)
    serve.add_argument("--port", type=int, default=8000)

    return parser


def main(argv: list[str] | None = None) -> int:
    load_project_env()
    args = _build_parser().parse_args(argv)

    if args.cmd == "up":
        if os.environ.get("RUN_INTERP_POD") != "1":
            print("Set RUN_INTERP_POD=1 to create the (billable) interp pod.", file=sys.stderr)
            return 1
        try:
            result, deadline = bring_up(
                model_name=args.model,
                gpu_id=args.gpu_id,
                image=args.image,
                container_disk_gb=args.container_disk_gb,
                hours=args.hours,
                now=datetime.now(UTC),
                env=os.environ,
                run=_subprocess_run,
            )
        except RuntimeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(result)
        print(
            f"# pod auto-stops at {deadline}. Teardown: "
            f"uv run python scripts/interp_pod.py down <pod_id>",
            file=sys.stderr,
        )
        return 0

    if args.cmd == "status":
        print(interp_pod.status(run=_subprocess_run))
        return 0

    if args.cmd == "stop":
        print(interp_pod.stop(args.pod_id, run=_subprocess_run))
        return 0

    if args.cmd == "down":
        print(interp_pod.terminate(args.pod_id, run=_subprocess_run))
        return 0

    if args.cmd == "serve-cmd":
        model = agent_model(args.model)
        # gpu_id/image are irrelevant to the serve line; placeholders keep the
        # spec constructable without demanding real ids for a print-only command.
        pod = InterpPodSpec(name=f"interp-{args.model}", gpu_id="-", image="-", http_port=args.port)
        print(" ".join(interp_pod.vllm_serve_command(model, pod)))
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
