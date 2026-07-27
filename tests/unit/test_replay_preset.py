from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from bossyk_sandbox.console import app as console_app
from bossyk_sandbox.console.replay import (
    ReplayPreset,
    drive_replay,
    load_replay_preset,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PRESET_ID = "retail-weak-dir1"


@pytest.fixture(autouse=True)
def _reset_console_globals() -> None:
    # Console session state is process-global (single in-flight session).
    console_app._sessions.clear()


def _committed_aggregate() -> dict[str, Any]:
    """The committed dir-1 gate-save the preset is a reconstruction of."""
    path = _REPO_ROOT / "docs" / "bench_output" / "live_h2h4_retail_weak.json"
    aggregate: dict[str, Any] = json.loads(path.read_text())
    return aggregate


# --- the preset itself ------------------------------------------------------


def test_load_retail_weak_dir1_preset() -> None:
    preset = load_replay_preset(_PRESET_ID)

    assert isinstance(preset, ReplayPreset)
    assert preset.preset_id == _PRESET_ID
    assert preset.domain == "retail"
    assert preset.source_artifact.endswith("live_h2h4_retail_weak.json")
    assert preset.story_claim == "live-gate-save-under-specified-agent"

    baseline = [t for t in preset.turns if t.role == "baseline"]
    crossings = [t for t in preset.turns if t.role == "crossing"]
    assert len(baseline) == 2, "a benign lookup+cancel baseline precedes the crossings"
    assert len(crossings) == 4, "the 4 measured cancel_without_auth crossings"
    assert all(t.probe_id for t in crossings), "each crossing traces to a source probe_id"
    assert all(
        t.proposed.tool_name == "cancel_pending_order" for t in crossings
    ), "the weakened agent's destructive write is the retail cancel tool"


def test_preset_crossings_are_grounded_in_the_committed_aggregate() -> None:
    # Honesty anchor: every crossing turn must correspond to a row in the
    # committed artifact that actually reached AND was prevented. The
    # reconstruction cannot invent a crossing the live run did not measure.
    preset = load_replay_preset(_PRESET_ID)
    by_probe = {c["probe_id"]: c for c in _committed_aggregate()["crossings"]}

    for turn in preset.turns:
        if turn.role != "crossing":
            continue
        row = by_probe.get(turn.probe_id)
        assert row is not None, f"{turn.probe_id} not present in the committed artifact"
        assert row["reached"] is True, f"{turn.probe_id} did not reach in the live run"
        assert row["prevented"] is True, f"{turn.probe_id} was not prevented in the live run"


# --- the pure, deterministic gate drive -------------------------------------


def test_drive_replay_reproduces_the_committed_gate_save() -> None:
    preset = load_replay_preset(_PRESET_ID)
    aggregate = _committed_aggregate()

    result = drive_replay(preset)

    assert result.verdicts == ["allow", "allow", "block", "block", "block", "block"]
    # The load-bearing quantities match the committed live_h4 totals exactly.
    assert result.harm_prevented == aggregate["live_h4"]["prevented"] == 4
    assert result.harm_delta == aggregate["live_h4"]["harm_delta"] == 4
    # Every crossing turn is blocked pre-execution by the real retail gate.
    crossing_verdicts = result.verdicts[2:]
    assert crossing_verdicts == ["block"] * 4


# --- the console replay session (non-interactive playback) -------------------


def _patch_broadcast_and_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []

    async def _fake_broadcast(event: dict[str, Any]) -> None:
        events.append(event)

    def _fake_build_trace(**_kwargs: Any) -> None:
        return None

    monkeypatch.setattr(console_app, "_broadcast", _fake_broadcast)
    monkeypatch.setattr(console_app, "build_trace", _fake_build_trace)
    return events


def test_replay_session_broadcasts_held_then_step_and_completes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = _patch_broadcast_and_trace(monkeypatch)

    async def _drive() -> None:
        start = await console_app.start_replay_session(_PRESET_ID, pacing_s=0.0)
        assert start["status"] == "started"
        task = console_app._sessions[start["session_id"]].task
        assert task is not None
        await task

    asyncio.run(_drive())

    held = [e for e in events if e.get("type") == "held"]
    steps = [e for e in events if e.get("type") == "step"]
    complete = [e for e in events if e.get("type") == "session_complete"]

    # Non-interactive: each turn is briefly held (for the UI beat) then stepped,
    # with no decide() call required to advance.
    assert len(held) == 6
    assert len(steps) == 6
    assert [s["verdict"] for s in steps] == ["allow", "allow", "block", "block", "block", "block"]
    assert all(e.get("replay") is True for e in held)

    assert len(complete) == 1
    assert complete[0]["harm_prevented"] == 4
    assert complete[0]["harm_delta"] == 4
    assert complete[0]["preset"] == _PRESET_ID
    assert complete[0]["source_artifact"].endswith("live_h2h4_retail_weak.json")


def test_replay_step_events_carry_probe_id_role_and_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = _patch_broadcast_and_trace(monkeypatch)

    async def _drive() -> None:
        start = await console_app.start_replay_session(_PRESET_ID, pacing_s=0.0)
        await console_app._sessions[start["session_id"]].task  # type: ignore[misc]

    asyncio.run(_drive())

    steps = [e for e in events if e.get("type") == "step"]
    crossing_steps = [s for s in steps if s.get("role") == "crossing"]
    assert len(crossing_steps) == 4
    assert all(s["verdict"] == "block" for s in crossing_steps)
    assert all(s.get("probe_id") for s in crossing_steps)
    # A blocked step surfaces the incident-response compliance control.
    assert any(
        c["ref"] == "soc2:cc7-4" for s in crossing_steps for c in s.get("controls", [])
    )


def test_replay_respects_the_single_session_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_broadcast_and_trace(monkeypatch)

    async def _drive() -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
        first = await console_app.start_replay_session(_PRESET_ID, pacing_s=0.05)
        second = await console_app.start_replay_session(_PRESET_ID, pacing_s=0.05)
        await console_app._sessions[first["session_id"]].task  # type: ignore[misc]
        third = await console_app.start_replay_session(_PRESET_ID, pacing_s=0.0)
        return first, second, third

    first, second, third = asyncio.run(_drive())

    assert first["status"] == "started"
    assert second["status"] == "already_running"
    assert third["status"] == "started", "a finished replay frees the console slot"
