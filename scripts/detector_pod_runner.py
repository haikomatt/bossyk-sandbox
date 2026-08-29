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
# Live-checked against `runpodctl pod create --help` on 2026-08-29
# (runpodctl 2.12.0-51ca7f0): NEITHER `--cost` NOR `--terminate-after` exists
# as a flag. The prior real attempt's script carried both anyway (dead
# flags -- `runpodctl` errors on unknown flags, so any invocation with them
# would fail outright before ever reaching the API); do not resurface them.
# Real-infra backstops against a runaway pod are, in order: the in-script
# `pod_budget` wall/dollar caps (checked every poll, see `should_terminate`),
# the supervisor's `finally`-block terminate + verified-empty-`pod list`
# check, and operator foreground supervision for the run's whole lifetime.
#
# The lost CA-MTL-1 pod (2026-08-28/29 attempt) motivates preferring a
# different secure-cloud region when `runpodctl gpu list` exposes the
# choice: for A40/secure-cloud on 2026-08-29, CA-MTL-1 showed "Low" stock
# (the region that vanished), EU-SE-1 showed "Medium", US-MO-1 showed
# "none". EU-SE-1 is used as the preferred `--data-center-ids` value; if a
# future check shows it unavailable, omit the flag and let RunPod's own
# placement choose (never silently fall back to CA-MTL-1 by name).
PREFERRED_DATA_CENTER_ID = "EU-SE-1"


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
    commands: list[str] = field(default_factory=list)  # every ssh_run command, in order

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
        self.commands.append(command)
        if self.ssh_script:
            return self.ssh_script.pop(0)
        return (0, "", "")

    def send_path(self, pod_id: str, local_path: Path, remote_path: str) -> None:
        self.sent.append((pod_id, local_path, remote_path))

    def receive_path(self, pod_id: str, remote_path: str, local_path: Path) -> None:
        self.received.append((pod_id, remote_path, local_path))


def _create_pod_args(*, name: str, gpu_type: str, data_center_ids: str | None = None) -> list[str]:
    """Pure argument-list builder for `runpodctl pod create` -- unit-testable
    without a subprocess, same pattern as `_remote_train_command`. Deliberately
    does NOT include `--cost` or `--terminate-after` (see the module-level
    comment by `PREFERRED_DATA_CENTER_ID`: neither flag exists in the
    installed runpodctl). `data_center_ids` is optional so a caller can omit
    region pinning entirely (falls back to RunPod's own placement) rather
    than hardcode a region that later stops being offered."""
    args = [
        "pod",
        "create",
        "--name",
        name,
        "--gpu-id",
        gpu_type,
        "--template-id",
        TRAINING_TEMPLATE_ID,
        "--cloud-type",
        "SECURE",
        "--container-disk-in-gb",
        "50",
        "--ports",
        "22/tcp",
    ]
    if data_center_ids:
        args += ["--data-center-ids", data_center_ids]
    return args


class RunpodCTLClient:
    """Real `PodClient`: `runpodctl pod ...` (the modern, non-deprecated
    subcommand form -- `runpodctl create pod` is deprecated and takes a
    different, incompatible flag set) for pod lifecycle, plain `ssh`/`scp`
    for command execution and file transfer (connection info + identity
    file from `runpodctl ssh info <pod-id>`: fields verified live against a
    real pod as `ip`/`port`/`ssh_key.path`, NOT `host`). Assumes
    `RUNPOD_API_KEY` is already in the environment (see module docstring) --
    never reads or writes any `.env` file itself."""

    def _runpodctl(self, *args: str) -> dict[str, Any]:
        proc = subprocess.run(
            ["runpodctl", *args, "-o", "json"], capture_output=True, text=True, timeout=60
        )
        if proc.returncode != 0:
            raise RuntimeError(f"runpodctl {' '.join(args)} failed: {proc.stderr.strip()}")
        return json.loads(proc.stdout) if proc.stdout.strip() else {}

    def create_pod(self, *, name: str, gpu_type: str) -> str:
        result = self._runpodctl(
            *_create_pod_args(
                name=name, gpu_type=gpu_type, data_center_ids=PREFERRED_DATA_CENTER_ID
            )
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

    def _ssh_target(self, pod_id: str) -> tuple[str, str, str]:
        """Returns `(ip, port, identity_file)` parsed from
        `runpodctl ssh info <pod_id>`. Raises if SSH isn't up yet -- the
        caller is expected to have already waited for readiness."""
        proc = subprocess.run(
            ["runpodctl", "ssh", "info", pod_id, "-o", "json"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"runpodctl ssh info {pod_id} failed: {proc.stderr.strip()}")
        info = json.loads(proc.stdout)
        if info.get("error"):
            raise RuntimeError(f"pod {pod_id} SSH not ready: {info['error']}")
        return str(info["ip"]), str(info.get("port", 22)), str(info["ssh_key"]["path"])

    def ssh_run(
        self, pod_id: str, command: str, *, timeout: float | None = None
    ) -> tuple[int, str, str]:
        ip, port, identity = self._ssh_target(pod_id)
        proc = subprocess.run(
            [
                "ssh",
                "-i",
                identity,
                "-p",
                port,
                "-o",
                "StrictHostKeyChecking=accept-new",
                f"root@{ip}",
                command,
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return proc.returncode, proc.stdout, proc.stderr

    def send_path(self, pod_id: str, local_path: Path, remote_path: str) -> None:
        ip, port, identity = self._ssh_target(pod_id)
        subprocess.run(
            [
                "ssh",
                "-i",
                identity,
                "-p",
                port,
                "-o",
                "StrictHostKeyChecking=accept-new",
                f"root@{ip}",
                f"mkdir -p {remote_path}",
            ],
            check=True,
            timeout=60,
        )
        subprocess.run(
            [
                "scp",
                "-i",
                identity,
                "-P",
                port,
                "-r",
                f"{local_path}/.",
                f"root@{ip}:{remote_path}/",
            ],
            check=True,
            timeout=1800,
        )

    def receive_path(self, pod_id: str, remote_path: str, local_path: Path) -> None:
        ip, port, identity = self._ssh_target(pod_id)
        local_path.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["scp", "-i", identity, "-P", port, "-r", f"root@{ip}:{remote_path}", str(local_path)],
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


def _remote_train_command(
    *,
    hard_cap_usd: float,
    max_wall_seconds: int,
    families: str | None = None,
    domains: str | None = None,
    seeds: str | None = None,
) -> str:
    """The command run on the pod (inside tmux): install the
    `detector-train` deps directly via pip (NOT `uv sync` -- the base
    project's REQUIRED deps include `auditk`/`tau2` as editable path deps
    on sibling checkouts that `bossyk_sandbox.detector.*` never actually
    imports; the `runpod-torch-v280` template already ships torch 2.8+cu128,
    so only `transformers`/`peft`/`accelerate` are missing -- see this run's
    lifecycle-log decision note), run the all-cells orchestrator with the
    SAME budget numbers as this local supervisor via `PYTHONPATH=src`, then
    drop `DONE_MARKER`.

    `families`/`domains`/`seeds` are optional pass-throughs to
    `detector_pod_train_all.py`'s own (comma-separated) cell-selection
    flags -- used for top-up runs that must re-train only specific cells
    (e.g. ones that died mid-sweep). Omitted by default, which reproduces
    the original full-sweep command unchanged: cell selection is opt-in,
    layered on top of (not a replacement for) the orchestrator's own
    manifest-resume skip logic, so a top-up run can never drift into
    training cells outside the caller's explicit selection."""
    cell_selection = ""
    if families is not None:
        cell_selection += f" --families {families}"
    if domains is not None:
        cell_selection += f" --domains {domains}"
    if seeds is not None:
        cell_selection += f" --seeds {seeds}"
    train_cmd = (
        f"cd {REMOTE_WORKDIR} && "
        "pip install -q --break-system-packages "
        "transformers peft accelerate sentencepiece protobuf && "
        f"PYTHONPATH={REMOTE_WORKDIR}/src python3 scripts/detector_pod_train_all.py "
        "--checkpoint-root /workspace/checkpoints "
        f"--manifest-out {REMOTE_WORKDIR}/probes/detector/results/training_manifest.jsonl "
        f"--scores-root {REMOTE_WORKDIR}/probes/detector/results/scores "
        f"--hard-cap-usd {hard_cap_usd} --max-wall-seconds {max_wall_seconds}"
        f"{cell_selection} "
        f"; echo $? > {REMOTE_WORKDIR}/{DONE_MARKER}"
    )
    session = "detector-train"
    return (
        f"which tmux || (apt-get update && apt-get install -y tmux); "
        f"tmux new-session -d -s {session} {shlex.quote(train_cmd)}"
    )


def _local_results_dir(local_dir: Path) -> Path:
    return local_dir / "probes" / "detector" / "results_from_pod"


def _sync_incremental(client: PodClient, pod_id: str, local_dir: Path, log: LifecycleLog) -> None:
    """Best-effort incremental download of the pod's results dir (manifest +
    per-cell score files) into a local staging dir. Called on every poll
    iteration the job is still running (plus once more right before any
    terminate) so a pod that vanishes mid-run costs at most the one
    in-flight cell -- the lesson from the prior attempt, which only
    downloaded once at the very end and lost 13 already-completed cells
    with the pod. Failures (e.g. a transient scp hiccup, or the pod already
    gone) are swallowed and logged, never fatal -- the next successful sync
    catches up, and if the pod really is gone, the last successful sync is
    what's left to recover."""
    try:
        client.receive_path(
            pod_id,
            f"{REMOTE_WORKDIR}/probes/detector/results",
            _local_results_dir(local_dir),
        )
    except Exception as exc:  # noqa: BLE001 -- best-effort; must never crash the poll loop
        log.poll_log.append(f"incremental sync failed (will retry next poll): {exc}")
        return
    manifest = _local_results_dir(local_dir) / "results" / "training_manifest.jsonl"
    n_cells = 0
    if manifest.exists():
        n_cells = sum(1 for line in manifest.read_text().splitlines() if line.strip())
    log.poll_log.append(f"incremental sync: {n_cells} cell(s) captured so far")


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
    families: str | None = None,
    domains: str | None = None,
    seeds: str | None = None,
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
            hard_cap_usd=hard_cap_usd,
            max_wall_seconds=max_wall_seconds,
            families=families,
            domains=domains,
            seeds=seeds,
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
                # one last best-effort pull -- catches anything landed since
                # the previous poll, before the pod goes away for good.
                _sync_incremental(client, pod_id, local_dir, log)
                _terminate_and_verify(decision.reason)
                return log

            check_rc, check_out, _ = client.ssh_run(
                pod_id, f"cat {REMOTE_WORKDIR}/{DONE_MARKER} 2>/dev/null || true", timeout=30
            )
            if check_out.strip():
                log.poll_log.append(f"remote job signalled done: exit={check_out.strip()}")
                _sync_incremental(client, pod_id, local_dir, log)
                _terminate_and_verify("remote job completed")
                return log

            log.poll_log.append(f"poll at {elapsed:.0f}s: not done yet")
            _sync_incremental(client, pod_id, local_dir, log)
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
    parser.add_argument(
        "--families",
        default=None,
        help="Comma-separated pass-through to detector_pod_train_all.py's "
        "--families (top-up runs only; omit for a full sweep).",
    )
    parser.add_argument(
        "--domains",
        default=None,
        help="Comma-separated pass-through to detector_pod_train_all.py's "
        "--domains (top-up runs only; omit for a full sweep).",
    )
    parser.add_argument(
        "--seeds",
        default=None,
        help="Comma-separated pass-through to detector_pod_train_all.py's "
        "--seeds (top-up runs only; omit for a full sweep).",
    )
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
        families=args.families,
        domains=args.domains,
        seeds=args.seeds,
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
