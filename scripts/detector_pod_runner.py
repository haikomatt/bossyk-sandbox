#!/usr/bin/env python
"""LOCAL supervisor for the detector-training pod run (phase-detector-
training.md build-order step 4, Phase B): provisions ONE A40-class RunPod
pod, uploads the repo + v2 corpus, launches `scripts/detector_pod_train_all.py`
inside a tmux session on the pod (interp-rigorous-run-runbook.md's lesson:
"Install tmux FIRST on any pod ... never rely on bare setsid detach"), polls
to completion or budget exhaustion, downloads results (manifests, metrics,
per-item scores -- NOT model weights), and ALWAYS terminates the pod in a
`finally` block, verified via a fresh `runpodctl pod list` afterward (brief:
"This verification is mandatory evidence, not optional.").

`RUNPOD_API_KEY` is read from the environment -- source it from the sibling
`bossyk-sandbox/.env` per the runbook's convention
(`export RUNPOD_API_KEY=$(grep ... | cut -d= -f2)`), never printed by this
script, never written to a new `.env` file.

Two layers of budget safety, both live in-script (never left to operator
judgement at call time):
- `bossyk_sandbox.detector.pod_budget` governs BOTH this script's poll loop
  (`--hard-cap-usd` / `--max-wall-seconds`) and the on-pod orchestrator's
  per-cell decisions -- same module, same numbers, no duplicated arithmetic.
- The `finally` block terminates the pod on every exit path: normal
  completion, budget breach, an exception, or Ctrl-C (SIGINT is also
  trapped so an operator abort still tears the pod down).

`PodClient` is a `Protocol` so the lifecycle state machine (provision -> poll
-> terminate -> verify) is hermetically testable against `FakePodClient`
(`tests/unit/test_detector_pod_runner.py`) without a real pod or API key.
`RunpodCTLClient` is the real backend, a thin subprocess wrapper around
`runpodctl` + plain `ssh`/`scp` -- NOT unit tested here (it is the one part
of this build that only a live pod can validate; the first real invocation
IS its integration test, run supervised, per house convention for
infra scripts).
"""

from __future__ import annotations

import argparse
import json
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from bossyk_sandbox.detector.pod_budget import (
    HARD_CAP_USD,
    MAX_WALL_SECONDS,
    TerminationDecision,
    should_terminate,
)

REPO_ROOT = Path(__file__).parent.parent
A40_GPU_TYPE = "NVIDIA A40"
TRAINING_TEMPLATE_ID = "runpod-torch-v280"  # torch 2.8, matches the house runbook convention
POLL_INTERVAL_SECONDS = 30
REMOTE_WORKDIR = "/workspace/bossyk-sandbox-partb"
DONE_MARKER = "DETECTOR_TRAIN_ALL_DONE"


class PodClient(Protocol):
    def create_pod(self, *, name: str, gpu_type: str) -> str:
        """Returns the new pod's id."""
        ...

    def get_status(self, pod_id: str) -> str:
        """e.g. 'RUNNING', 'EXITED', 'CREATED'."""
        ...

    def list_pod_ids(self) -> list[str]:
        """Every pod id currently on the account (any status the CLI's
        default listing shows -- used for the mandatory post-terminate
        verification, so a lingering pod in any non-terminated state is
        caught)."""
        ...

    def terminate_pod(self, pod_id: str) -> None: ...

    def ssh_run(
        self, pod_id: str, command: str, *, timeout: float | None = None
    ) -> tuple[int, str, str]:
        """Runs `command` on the pod over SSH; returns (returncode, stdout, stderr)."""
        ...

    def send_path(self, pod_id: str, local_path: Path, remote_path: str) -> None: ...

    def receive_path(self, pod_id: str, remote_path: str, local_path: Path) -> None: ...


@dataclass
class FakePodClient:
    """Hermetic double for `PodClient` -- an in-memory pod registry, no
    subprocess/network. Used by `tests/unit/test_detector_pod_runner.py` to
    prove the lifecycle state machine (provision -> poll -> terminate ->
    verify-0-pods) always tears down, on every exit path, without needing a
    real RunPod account."""

    pods: dict[str, str] = field(default_factory=dict)  # pod_id -> status
    ssh_script: list[tuple[int, str, str]] = field(default_factory=list)  # scripted ssh_run replies
    _next_id: int = 0
    sent: list[tuple[str, Path, str]] = field(default_factory=list)
    received: list[tuple[str, str, Path]] = field(default_factory=list)
    terminated: list[str] = field(default_factory=list)

    def create_pod(self, *, name: str, gpu_type: str) -> str:
        self._next_id += 1
        pod_id = f"fake-pod-{self._next_id}"
        self.pods[pod_id] = "RUNNING"
        return pod_id

    def get_status(self, pod_id: str) -> str:
        return self.pods.get(pod_id, "EXITED")

    def list_pod_ids(self) -> list[str]:
        return [pid for pid, status in self.pods.items() if status != "EXITED"]

    def terminate_pod(self, pod_id: str) -> None:
        self.pods[pod_id] = "EXITED"
        self.terminated.append(pod_id)

    def ssh_run(
        self, pod_id: str, command: str, *, timeout: float | None = None
    ) -> tuple[int, str, str]:
        if self.ssh_script:
            return self.ssh_script.pop(0)
        return (0, "", "")

    def send_path(self, pod_id: str, local_path: Path, remote_path: str) -> None:
        self.sent.append((pod_id, local_path, remote_path))

    def receive_path(self, pod_id: str, remote_path: str, local_path: Path) -> None:
        self.received.append((pod_id, remote_path, local_path))


class RunpodCTLClient:
    """Real `PodClient`: `runpodctl` for pod lifecycle, plain `ssh`/`scp` for
    command execution and file transfer (connection info from
    `runpodctl ssh info <pod-id>`). Assumes `RUNPOD_API_KEY` is already in
    the environment (see module docstring) -- never reads or writes any
    `.env` file itself."""

    def _runpodctl(self, *args: str) -> dict[str, Any]:
        proc = subprocess.run(
            ["runpodctl", *args, "-o", "json"], capture_output=True, text=True, timeout=60
        )
        if proc.returncode != 0:
            raise RuntimeError(f"runpodctl {' '.join(args)} failed: {proc.stderr.strip()}")
        return json.loads(proc.stdout) if proc.stdout.strip() else {}

    def create_pod(self, *, name: str, gpu_type: str) -> str:
        result = self._runpodctl(
            "create",
            "pod",
            "--name",
            name,
            "--gpuType",
            gpu_type,
            "--templateId",
            TRAINING_TEMPLATE_ID,
            "--containerDiskSize",
            "50",
            "--ports",
            "22/tcp",
            "--secureCloud",
            "--startSSH",
        )
        pod_id = result.get("id") or result.get("podId")
        if not pod_id:
            raise RuntimeError(f"could not find pod id in create response: {result!r}")
        return str(pod_id)

    def get_status(self, pod_id: str) -> str:
        result = self._runpodctl("pod", "get", pod_id)
        status = result.get("desiredStatus") or result.get("status") or "UNKNOWN"
        return str(status)

    def list_pod_ids(self) -> list[str]:
        result = self._runpodctl("pod", "list", "--all")
        pods = result if isinstance(result, list) else result.get("pods", [])
        return [str(p["id"]) for p in pods if p.get("desiredStatus", p.get("status")) != "EXITED"]

    def terminate_pod(self, pod_id: str) -> None:
        subprocess.run(
            ["runpodctl", "pod", "delete", pod_id], capture_output=True, text=True, timeout=60
        )

    def _ssh_target(self, pod_id: str) -> tuple[str, str]:
        """Returns `(host, port)` parsed from `runpodctl ssh info <pod_id>`."""
        proc = subprocess.run(
            ["runpodctl", "ssh", "info", pod_id, "-o", "json"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"runpodctl ssh info {pod_id} failed: {proc.stderr.strip()}")
        info = json.loads(proc.stdout)
        return str(info["host"]), str(info.get("port", 22))

    def ssh_run(
        self, pod_id: str, command: str, *, timeout: float | None = None
    ) -> tuple[int, str, str]:
        host, port = self._ssh_target(pod_id)
        proc = subprocess.run(
            ["ssh", "-p", port, "-o", "StrictHostKeyChecking=accept-new", f"root@{host}", command],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return proc.returncode, proc.stdout, proc.stderr

    def send_path(self, pod_id: str, local_path: Path, remote_path: str) -> None:
        host, port = self._ssh_target(pod_id)
        subprocess.run(
            ["scp", "-P", port, "-r", str(local_path), f"root@{host}:{remote_path}"],
            check=True,
            timeout=1800,
        )

    def receive_path(self, pod_id: str, remote_path: str, local_path: Path) -> None:
        host, port = self._ssh_target(pod_id)
        local_path.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["scp", "-P", port, "-r", f"root@{host}:{remote_path}", str(local_path)],
            check=True,
            timeout=1800,
        )


@dataclass
class LifecycleLog:
    pod_id: str | None = None
    provisioned_at: float | None = None
    terminated_at: float | None = None
    terminate_reason: str = ""
    remaining_pods_after_terminate: list[str] = field(default_factory=list)
    verified_zero_pods: bool = False
    poll_log: list[str] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "pod_id": self.pod_id,
            "provisioned_at": self.provisioned_at,
            "terminated_at": self.terminated_at,
            "terminate_reason": self.terminate_reason,
            "remaining_pods_after_terminate": self.remaining_pods_after_terminate,
            "verified_zero_pods": self.verified_zero_pods,
            "poll_log": self.poll_log,
            "error": self.error,
        }


def _remote_train_command(*, hard_cap_usd: float, max_wall_seconds: int) -> str:
    """The command run on the pod (inside tmux): sync the `detector-train`
    extra, run the all-cells orchestrator with the SAME budget numbers as
    this local supervisor, then drop `DONE_MARKER`."""
    train_cmd = (
        f"cd {REMOTE_WORKDIR} && "
        "uv sync --extra detector-train && "
        "uv run python scripts/detector_pod_train_all.py "
        "--checkpoint-root /workspace/checkpoints "
        f"--manifest-out {REMOTE_WORKDIR}/probes/detector/results/training_manifest.jsonl "
        f"--scores-root {REMOTE_WORKDIR}/probes/detector/results/scores "
        f"--hard-cap-usd {hard_cap_usd} --max-wall-seconds {max_wall_seconds} "
        f"; echo $? > {REMOTE_WORKDIR}/{DONE_MARKER}"
    )
    session = "detector-train"
    return (
        f"which tmux || (apt-get update && apt-get install -y tmux); "
        f"tmux new-session -d -s {session} {shlex.quote(train_cmd)}"
    )


def run_lifecycle(
    client: PodClient,
    *,
    pod_name: str,
    gpu_type: str = A40_GPU_TYPE,
    hard_cap_usd: float = HARD_CAP_USD,
    max_wall_seconds: int = MAX_WALL_SECONDS,
    poll_interval_seconds: float = POLL_INTERVAL_SECONDS,
    local_dir: Path,
    dry_run_remote: bool = False,
    sleep_fn: Any = time.sleep,
    now_fn: Any = time.monotonic,
) -> LifecycleLog:
    """The full supervised lifecycle: provision, upload, launch, poll until
    the remote job signals `DONE_MARKER` or the budget runs out, download
    results, and ALWAYS terminate + verify in `finally` -- this function's
    `finally` block is the thing the brief calls "pod termination in a
    finally/trap path" and is exercised directly by the hermetic tests
    (budget breach, remote exception, KeyboardInterrupt) against
    `FakePodClient`."""
    log = LifecycleLog()
    start = now_fn()
    pod_id: str | None = None

    def _terminate_and_verify(reason: str) -> None:
        log.terminate_reason = reason
        if pod_id is not None:
            client.terminate_pod(pod_id)
            log.terminated_at = now_fn()
        log.remaining_pods_after_terminate = client.list_pod_ids()
        log.verified_zero_pods = len(log.remaining_pods_after_terminate) == 0
        if not log.verified_zero_pods:
            log.poll_log.append(
                f"WARNING: pods still listed after terminate: {log.remaining_pods_after_terminate}"
            )

    try:
        pod_id = client.create_pod(name=pod_name, gpu_type=gpu_type)
        log.pod_id = pod_id
        log.provisioned_at = now_fn()
        log.poll_log.append(f"provisioned {pod_id}")

        client.send_path(pod_id, local_dir, REMOTE_WORKDIR)
        log.poll_log.append(f"uploaded {local_dir} -> {REMOTE_WORKDIR}")

        remote_cmd = _remote_train_command(
            hard_cap_usd=hard_cap_usd, max_wall_seconds=max_wall_seconds
        )
        rc, out, err = client.ssh_run(pod_id, remote_cmd, timeout=120)
        log.poll_log.append(f"launched training session rc={rc}")
        if rc != 0:
            log.error = f"failed to launch remote training session: {err.strip()}"
            return log

        while True:
            elapsed = now_fn() - start
            decision: TerminationDecision = should_terminate(
                elapsed, hard_cap_usd=hard_cap_usd, max_wall_seconds=max_wall_seconds
            )
            if decision.terminate:
                log.poll_log.append(f"budget backstop fired: {decision.reason}")
                _terminate_and_verify(decision.reason)
                return log

            check_rc, check_out, _ = client.ssh_run(
                pod_id, f"cat {REMOTE_WORKDIR}/{DONE_MARKER} 2>/dev/null || true", timeout=30
            )
            if check_out.strip():
                log.poll_log.append(f"remote job signalled done: exit={check_out.strip()}")
                client.receive_path(
                    pod_id,
                    f"{REMOTE_WORKDIR}/probes/detector/results",
                    local_dir / "probes" / "detector" / "results_from_pod",
                )
                _terminate_and_verify("remote job completed")
                return log

            log.poll_log.append(f"poll at {elapsed:.0f}s: not done yet")
            if dry_run_remote:
                # test/dry-run mode: don't actually sleep in a loop forever
                _terminate_and_verify("dry_run_remote poll-once")
                return log
            sleep_fn(poll_interval_seconds)

    except BaseException as exc:  # noqa: BLE001 -- deliberately broad: ANY exit path must terminate
        log.error = f"{type(exc).__name__}: {exc}"
        _terminate_and_verify(f"exception during lifecycle: {log.error}")
        raise
    finally:
        # Belt-and-braces: even if a bug above returns without calling
        # _terminate_and_verify, this still guarantees a termination attempt
        # and a fresh listing before the function actually exits.
        if pod_id is not None and log.terminated_at is None:
            _terminate_and_verify("finally-path safety net (should be unreachable)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--pod-name", default="detector-training")
    parser.add_argument("--gpu-type", default=A40_GPU_TYPE)
    parser.add_argument("--hard-cap-usd", type=float, default=HARD_CAP_USD)
    parser.add_argument("--max-wall-seconds", type=int, default=MAX_WALL_SECONDS)
    parser.add_argument("--local-dir", type=Path, default=REPO_ROOT)
    parser.add_argument("--summary-out", type=Path, default=None)
    args = parser.parse_args(argv)

    client: PodClient = RunpodCTLClient()

    def _sigint_handler(signum: int, frame: Any) -> None:
        raise KeyboardInterrupt()

    signal.signal(signal.SIGINT, _sigint_handler)
    signal.signal(signal.SIGTERM, _sigint_handler)

    log = run_lifecycle(
        client,
        pod_name=args.pod_name,
        gpu_type=args.gpu_type,
        hard_cap_usd=args.hard_cap_usd,
        max_wall_seconds=args.max_wall_seconds,
        local_dir=args.local_dir,
    )

    if args.summary_out:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(json.dumps(log.to_dict(), indent=2))

    print(json.dumps(log.to_dict(), indent=2), file=sys.stderr)
    if not log.verified_zero_pods:
        print("FATAL: could not verify 0 pods remain after terminate", file=sys.stderr)
        return 1
    return 0 if log.error is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
