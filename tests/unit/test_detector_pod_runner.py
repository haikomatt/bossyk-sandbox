"""Hermetic tests for scripts/detector_pod_runner.py's lifecycle state
machine (provision -> poll -> terminate -> verify), against FakePodClient --
no subprocess, no network, no RunPod account. This is the safety-critical
part of Phase B: proving the pod is ALWAYS terminated (success, budget
breach, remote failure, exception) and that termination is verified via a
fresh listing, not assumed."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

SCRIPT_PATH = Path(__file__).parent.parent.parent / "scripts" / "detector_pod_runner.py"


def _import() -> ModuleType:
    spec = importlib.util.spec_from_file_location("detector_pod_runner_script", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    # dataclasses' string-annotation resolution looks the module up via
    # `sys.modules[cls.__module__]` -- register it before exec_module (this
    # script is the first importlib-loaded script to define its own
    # dataclasses; the sibling scripts only import already-registered ones).
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


def _noop_sleep(_seconds: float) -> None:
    return None


def test_lifecycle_completes_and_terminates_on_remote_done_signal(tmp_path: Path) -> None:
    m = _import()
    client = m.FakePodClient()
    # launch rc=0, then one "not done" poll, then "done" with exit 0.
    client.ssh_script = [
        (0, "", ""),  # launch
        (0, "", ""),  # first done-check: empty -> not done
        (0, "0\n", ""),  # second done-check: DONE_MARKER contents "0"
    ]
    times = iter([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])

    def fake_now() -> float:
        return next(times, 100.0)

    log = m.run_lifecycle(
        client,
        pod_name="test-pod",
        local_dir=tmp_path,
        sleep_fn=_noop_sleep,
        now_fn=fake_now,
    )

    assert log.pod_id is not None
    assert log.error is None
    assert log.terminated_at is not None
    assert log.verified_zero_pods is True
    assert log.remaining_pods_after_terminate == []
    assert client.terminated == [log.pod_id]
    assert client.list_pod_ids() == []


def test_lifecycle_terminates_on_budget_breach_without_ever_finishing(tmp_path: Path) -> None:
    m = _import()
    client = m.FakePodClient()
    client.ssh_script = [(0, "", "")]  # launch ok; then budget fires before any done-check matters

    log = m.run_lifecycle(
        client,
        pod_name="test-pod",
        local_dir=tmp_path,
        hard_cap_usd=0.0001,
        max_wall_seconds=10**9,
        sleep_fn=_noop_sleep,
        now_fn=lambda: 3600.0,  # "elapsed" always looks huge relative to the tiny cap
    )

    assert log.pod_id is not None
    assert "budget" in log.terminate_reason or "cap" in log.terminate_reason
    assert log.verified_zero_pods is True
    assert client.terminated == [log.pod_id]


def test_lifecycle_terminates_when_remote_launch_fails(tmp_path: Path) -> None:
    m = _import()
    client = m.FakePodClient()
    client.ssh_script = [(1, "", "tmux: command not found")]  # launch fails

    log = m.run_lifecycle(client, pod_name="test-pod", local_dir=tmp_path, sleep_fn=_noop_sleep)

    assert log.error is not None
    assert "failed to launch" in log.error
    assert log.verified_zero_pods is True
    assert client.terminated == [log.pod_id]


def test_lifecycle_terminates_on_exception_and_reraises(tmp_path: Path) -> None:
    m = _import()

    class RaisingClient(m.FakePodClient):  # type: ignore[name-defined,misc]
        def ssh_run(
            self, pod_id: str, command: str, *, timeout: float | None = None
        ) -> tuple[int, str, str]:
            raise RuntimeError("ssh connection reset")

    client = RaisingClient()
    try:
        m.run_lifecycle(client, pod_name="test-pod", local_dir=tmp_path, sleep_fn=_noop_sleep)
        raise AssertionError("expected RuntimeError to propagate")
    except RuntimeError as exc:
        assert "ssh connection reset" in str(exc)

    assert len(client.terminated) == 1
    assert client.list_pod_ids() == []


def test_lifecycle_flags_when_termination_verification_fails(tmp_path: Path) -> None:
    m = _import()

    class StubbornClient(m.FakePodClient):  # type: ignore[name-defined,misc]
        def terminate_pod(self, pod_id: str) -> None:
            self.terminated.append(pod_id)
            # deliberately does NOT clear status -- simulates a pod that
            # refuses to die, so `list_pod_ids` still reports it.

    client = StubbornClient()
    client.ssh_script = [(1, "", "boom")]  # fail fast so we reach the finally quickly

    log = m.run_lifecycle(client, pod_name="test-pod", local_dir=tmp_path, sleep_fn=_noop_sleep)

    assert log.verified_zero_pods is False
    assert log.pod_id in log.remaining_pods_after_terminate
    assert any("WARNING" in line for line in log.poll_log)


def test_lifecycle_uploads_before_launching(tmp_path: Path) -> None:
    m = _import()
    client = m.FakePodClient()
    client.ssh_script = [(1, "", "boom")]  # fail fast
    log = m.run_lifecycle(client, pod_name="test-pod", local_dir=tmp_path, sleep_fn=_noop_sleep)
    assert len(client.sent) == 1
    assert client.sent[0][0] == log.pod_id
    assert client.sent[0][1] == tmp_path


def test_remote_train_command_embeds_the_same_budget_numbers() -> None:
    m = _import()
    cmd = m._remote_train_command(hard_cap_usd=12.5, max_wall_seconds=1800)
    assert "--hard-cap-usd 12.5" in cmd
    assert "--max-wall-seconds 1800" in cmd
    assert "detector_pod_train_all.py" in cmd


def test_remote_train_command_omits_cell_selection_flags_by_default() -> None:
    """Default (no families/domains/seeds passed) must reproduce the
    original full-sweep command byte-for-byte -- top-up/cell-selection runs
    are opt-in, never accidentally narrower than a full run."""
    m = _import()
    cmd = m._remote_train_command(hard_cap_usd=12.5, max_wall_seconds=1800)
    assert "--families" not in cmd
    assert "--domains" not in cmd
    assert "--seeds" not in cmd


def test_remote_train_command_forwards_cell_selection_flags_when_given() -> None:
    """A top-up run (e.g. re-running only the two cells that died to the
    disk-full bug) must be able to restrict the on-pod orchestrator to
    exactly those cells, independent of manifest-resume behaviour -- so the
    pod can never drift into training cells outside the authorized set."""
    m = _import()
    cmd = m._remote_train_command(
        hard_cap_usd=2.0,
        max_wall_seconds=1800,
        families="qwen_lora",
        domains="advice-eligibility",
        seeds="1,2",
    )
    assert "--families qwen_lora" in cmd
    assert "--domains advice-eligibility" in cmd
    assert "--seeds 1,2" in cmd
    # must appear before the exit-code capture, i.e. actually part of the
    # detector_pod_train_all.py invocation, not tacked on after it
    assert cmd.index("--seeds 1,2") < cmd.index("echo $?")


def test_create_pod_args_omit_unsupported_cost_and_terminate_after_flags() -> None:
    """runpodctl 2.12.0's `pod create --help` (checked live 2026-08-29) has
    neither `--cost` nor `--terminate-after` -- the prior attempt's script
    carried both as dead flags, which would make the real command fail
    outright with an unknown-flag error before ever reaching the API."""
    m = _import()
    args = m._create_pod_args(name="detector-training", gpu_type=m.A40_GPU_TYPE)
    assert "--cost" not in args
    assert "--terminate-after" not in args


def test_create_pod_args_prefer_a_non_ca_mtl_1_region_when_given() -> None:
    """The lost pod was CA-MTL-1 (Low stock); EU-SE-1 (Medium stock) is the
    preferred alternative -- dodging a possibly-flaky host fleet per the
    brief, when runpodctl exposes the choice via --data-center-ids."""
    m = _import()
    args = m._create_pod_args(
        name="detector-training", gpu_type=m.A40_GPU_TYPE, data_center_ids="EU-SE-1"
    )
    assert "--data-center-ids" in args
    assert "EU-SE-1" in args
    assert "CA-MTL-1" not in args


def test_create_pod_args_omit_region_flag_when_none_given() -> None:
    """Region pinning is optional -- omitting it falls back to RunPod's own
    placement rather than hardcoding a region that might stop being
    offered."""
    m = _import()
    args = m._create_pod_args(name="detector-training", gpu_type=m.A40_GPU_TYPE)
    assert "--data-center-ids" not in args


def test_create_pod_args_carry_the_core_provisioning_flags() -> None:
    m = _import()
    args = m._create_pod_args(name="detector-training", gpu_type=m.A40_GPU_TYPE)
    assert "--name" in args and "detector-training" in args
    assert "--gpu-id" in args and m.A40_GPU_TYPE in args
    assert "--template-id" in args and m.TRAINING_TEMPLATE_ID in args
    assert "--cloud-type" in args and "SECURE" in args
    assert "--container-disk-in-gb" in args and "50" in args


def test_lifecycle_syncs_incrementally_on_every_poll_not_just_at_completion(
    tmp_path: Path,
) -> None:
    """The prior real attempt (2026-08-29 Amendment-4 retrain) downloaded
    results only once, at the very end -- when the pod vanished mid-run it
    lost 13 already-completed cells with it. The poll loop must pull the
    pod's results dir (manifest + score files) down on every iteration that
    shows the job still running, not just once after DONE_MARKER, so a pod
    loss costs at most the one in-flight cell."""
    m = _import()
    client = m.FakePodClient()
    client.ssh_script = [
        (0, "", ""),  # launch
        (0, "", ""),  # poll 1: not done yet
        (0, "", ""),  # poll 2: not done yet
        (0, "", ""),  # poll 3: not done yet
        (0, "0\n", ""),  # poll 4: done
    ]
    log = m.run_lifecycle(client, pod_name="test-pod", local_dir=tmp_path, sleep_fn=_noop_sleep)

    assert log.error is None
    # one receive_path per not-done poll (3) plus the final completion sync (1)
    assert len(client.received) >= 4
    # every incremental sync pulls the SAME remote results dir into the SAME
    # local staging dir, so a later sync can only add to what an earlier one
    # already captured (no per-poll drift in the target path).
    remote_paths = {r[1] for r in client.received}
    local_paths = {r[2] for r in client.received}
    assert remote_paths == {f"{m.REMOTE_WORKDIR}/probes/detector/results"}
    assert local_paths == {tmp_path / "probes" / "detector" / "results_from_pod"}
    assert any("incremental sync" in line for line in log.poll_log)


def test_lifecycle_attempts_incremental_sync_before_terminating_on_budget_breach(
    tmp_path: Path,
) -> None:
    """A budget-exhaustion termination is also a "pod about to go away"
    moment from the results' point of view -- one last best-effort sync
    should be attempted before terminate, to catch anything landed since the
    previous poll."""
    m = _import()
    client = m.FakePodClient()
    client.ssh_script = [(0, "", "")]  # launch ok; budget fires before any done-check

    log = m.run_lifecycle(
        client,
        pod_name="test-pod",
        local_dir=tmp_path,
        hard_cap_usd=0.0001,
        max_wall_seconds=10**9,
        sleep_fn=_noop_sleep,
        now_fn=lambda: 3600.0,
    )

    assert log.verified_zero_pods is True
    assert len(client.received) >= 1


def test_lifecycle_incremental_sync_failure_is_swallowed_not_fatal(tmp_path: Path) -> None:
    """A transient scp hiccup mid-poll must not kill the whole supervisor --
    the next successful poll's sync catches up. Only a hard failure in the
    actual training-status check (ssh_run) should end the loop."""
    m = _import()

    class FlakySyncClient(m.FakePodClient):  # type: ignore[name-defined,misc]
        def receive_path(self, pod_id: str, remote_path: str, local_path: Path) -> None:
            raise RuntimeError("scp: connection reset by peer")

    client = FlakySyncClient()
    client.ssh_script = [
        (0, "", ""),  # launch
        (0, "", ""),  # poll 1: not done
        (0, "0\n", ""),  # poll 2: done
    ]
    log = m.run_lifecycle(client, pod_name="test-pod", local_dir=tmp_path, sleep_fn=_noop_sleep)

    assert log.error is None
    assert log.verified_zero_pods is True
    assert any("incremental sync failed" in line for line in log.poll_log)


def test_run_lifecycle_threads_cell_selection_into_the_remote_command(tmp_path: Path) -> None:
    """End-to-end: passing families/domains/seeds into run_lifecycle must
    reach the actual ssh_run command sent to the pod client (FakePodClient
    records every command it's asked to run in `.commands`)."""
    m = _import()
    client = m.FakePodClient()
    client.ssh_script = [
        (0, "", ""),  # launch
        (0, "DETECTOR_TRAIN_ALL_DONE\n0\n", ""),  # done marker check
    ]
    m.run_lifecycle(
        client,
        pod_name="test-pod",
        local_dir=tmp_path,
        sleep_fn=_noop_sleep,
        families="qwen_lora",
        domains="advice-eligibility",
        seeds="1,2",
    )
    launch_commands = [cmd for cmd in client.commands if "detector_pod_train_all.py" in cmd]
    assert len(launch_commands) == 1
    assert "--families qwen_lora" in launch_commands[0]
    assert "--domains advice-eligibility" in launch_commands[0]
    assert "--seeds 1,2" in launch_commands[0]
