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
