from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from bossyk_sandbox.console import app as console_app
from bossyk_sandbox.console.replay import drive_replay, load_replay_preset

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PRESET_ID = "retail-weak-modes"
_ALL_MODES = {"allow", "redirect", "defer", "step-up", "escalate"}


@pytest.fixture(autouse=True)
def _reset_console_globals() -> None:
    console_app._sessions.clear()


def _committed_aggregate() -> dict[str, Any]:
    path = _REPO_ROOT / "docs" / "bench_output" / "live_h2h4_retail_weak.json"
    aggregate: dict[str, Any] = json.loads(path.read_text())
    return aggregate


def _expected() -> dict[str, Any]:
    path = _REPO_ROOT / "story" / "replay" / f"{_PRESET_ID}.json"
    preset: dict[str, Any] = json.loads(path.read_text())
    expected: dict[str, Any] = preset["expected"]
    return expected


# --- the preset carries authored resolution modes ---------------------------


def test_modes_preset_exercises_the_full_taxonomy() -> None:
    preset = load_replay_preset(_PRESET_ID)

    modes = [turn.mode for turn in preset.turns]
    assert set(modes) == _ALL_MODES, "every resolution mode should appear at least once"
    assert modes == _expected()["modes"]


def test_redirect_turns_stay_grounded_in_the_committed_aggregate() -> None:
    # The two 'redirect' turns are the real dir-1 crossings; the authored
    # mode-demo turns (step-up / defer / escalate) must NOT claim a probe_id.
    preset = load_replay_preset(_PRESET_ID)
    by_probe = {c["probe_id"]: c for c in _committed_aggregate()["crossings"]}

    for turn in preset.turns:
        if turn.mode == "redirect":
            row = by_probe.get(turn.probe_id)
            assert row is not None, f"{turn.probe_id} not in the committed artifact"
            assert row["reached"] is True and row["prevented"] is True
        elif turn.mode in {"step-up", "defer", "escalate"}:
            assert turn.probe_id is None, "authored mode-demo turns are not grounded crossings"


# --- driving the preset emits modes + a hard-cell HITL queue -----------------


def test_drive_replay_emits_modes_and_orders_the_hitl_queue() -> None:
    preset = load_replay_preset(_PRESET_ID)
    expected = _expected()

    result = drive_replay(preset)

    # Structural gate verdicts stay real; modes are the authored overlay.
    assert result.verdicts == expected["verdicts"]
    assert result.modes == expected["modes"]

    # Only 'escalate' turns reach the hard-cell queue, ordered by severity.
    assert [item["severity"] for item in result.hitl_queue] == expected["hitl_order"]
    assert len(result.hitl_queue) == expected["hitl_count"] == 2
    for item in result.hitl_queue:
        assert item["reason"] and item["resolution"]
        assert item["tool_name"]


# --- the console broadcasts modes + the queue --------------------------------


def _patch_broadcast_and_trace(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []

    async def _fake_broadcast(event: dict[str, Any]) -> None:
        events.append(event)

    def _fake_build_trace(**_kwargs: Any) -> None:
        return None

    monkeypatch.setattr(console_app, "_broadcast", _fake_broadcast)
    monkeypatch.setattr(console_app, "build_trace", _fake_build_trace)
    return events


def test_console_replay_broadcasts_modes_and_the_hitl_queue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = _patch_broadcast_and_trace(monkeypatch)

    async def _drive() -> None:
        start = await console_app.start_replay_session(_PRESET_ID, pacing_s=0.0)
        assert start["status"] == "started"
        await console_app._sessions[start["session_id"]].task  # type: ignore[misc]

    asyncio.run(_drive())

    steps = [e for e in events if e.get("type") == "step"]
    complete = next(e for e in events if e.get("type") == "session_complete")

    assert [s["mode"] for s in steps] == _expected()["modes"]

    # Escalated steps carry their hard-cell payload inline.
    escalated = [s for s in steps if s["mode"] == "escalate"]
    assert len(escalated) == 2
    assert all(s.get("hitl") and s["hitl"]["severity"] for s in escalated)

    # session_complete surfaces the priority-ordered queue for the HITL panel.
    assert [item["severity"] for item in complete["hitl_queue"]] == ["critical", "high"]
