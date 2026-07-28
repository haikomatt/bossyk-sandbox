from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from bossyk_sandbox.console import app as console_app

# End-to-end, fully offline: drives the real console replay flow through the
# real endpoint (start_replay_session), the real session driver
# (_run_replay_session), and the real _broadcast fan-out over
# console_app._connections -- then checks the streamed gate-save against the
# committed live-run aggregate. The ONLY seam is the socket byte transport: a
# recording connection stands in for a websocket client, but it is registered
# in the real _connections set and receives via the real _broadcast loop (the
# unit test, by contrast, monkeypatches _broadcast out entirely). No network,
# no model calls, no local stack, so -- like test_live_h2h4_e2e /
# test_probe_grid_e2e -- it runs un-gated in the normal `pytest tests/` gate.

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PRESET_ID = "retail-weak-dir1"


class _RecordingConnection:
    """Stands in for a connected websocket client. `_broadcast` fans out to
    every object in `console_app._connections` by awaiting `send_json`, so this
    captures exactly the frames a real client would receive."""

    def __init__(self) -> None:
        self.frames: list[dict[str, Any]] = []

    async def send_json(self, event: dict[str, Any]) -> None:
        self.frames.append(event)


def _committed_aggregate() -> dict[str, Any]:
    path = _REPO_ROOT / "docs" / "bench_output" / "live_h2h4_retail_weak.json"
    aggregate: dict[str, Any] = json.loads(path.read_text())
    return aggregate


def test_console_replay_broadcasts_the_gate_save_end_to_end() -> None:
    async def _run() -> list[dict[str, Any]]:
        console_app._sessions.clear()
        console_app._connections.clear()
        client = _RecordingConnection()
        console_app._connections.add(client)  # type: ignore[arg-type]
        try:
            started = await console_app.start_replay_session(_PRESET_ID, pacing_s=0.0)
            assert started["status"] == "started"
            task = console_app._sessions[started["session_id"]].task
            assert task is not None
            await task
        finally:
            console_app._connections.discard(client)
        return client.frames

    events = asyncio.run(_run())

    held = [e for e in events if e["type"] == "held"]
    steps = [e for e in events if e["type"] == "step"]
    complete = next(e for e in events if e["type"] == "session_complete")

    # The full turn-by-turn sequence was broadcast to the connected client.
    assert len(held) == 6
    assert [s["verdict"] for s in steps] == ["allow", "allow", "block", "block", "block", "block"]

    # The streamed harm delta matches the committed live run exactly.
    aggregate = _committed_aggregate()
    assert complete["harm_prevented"] == aggregate["live_h4"]["prevented"] == 4
    assert complete["harm_delta"] == aggregate["live_h4"]["harm_delta"] == 4
    assert complete["preset"] == _PRESET_ID
    assert complete["source_artifact"].endswith("live_h2h4_retail_weak.json")

    # Every crossing was blocked pre-execution and carries its source probe_id.
    crossings = [s for s in steps if s["role"] == "crossing"]
    assert len(crossings) == 4
    assert all(s["verdict"] == "block" and s["probe_id"] for s in crossings)


def _run_replay(preset_id: str) -> list[dict[str, Any]]:
    async def _run() -> list[dict[str, Any]]:
        console_app._sessions.clear()
        console_app._connections.clear()
        client = _RecordingConnection()
        console_app._connections.add(client)  # type: ignore[arg-type]
        try:
            started = await console_app.start_replay_session(preset_id, pacing_s=0.0)
            assert started["status"] == "started"
            task = console_app._sessions[started["session_id"]].task
            assert task is not None
            await task
        finally:
            console_app._connections.discard(client)
        return client.frames

    return asyncio.run(_run())


def test_console_replay_broadcasts_modes_and_hitl_queue_end_to_end() -> None:
    events = _run_replay("retail-weak-modes")

    steps = [e for e in events if e["type"] == "step"]
    complete = next(e for e in events if e["type"] == "session_complete")

    # The full resolution-mode taxonomy streamed over the wire.
    assert [s["mode"] for s in steps] == [
        "allow",
        "allow",
        "redirect",
        "redirect",
        "step-up",
        "defer",
        "escalate",
        "escalate",
    ]
    # Real structural gate verdicts still ride alongside the authored modes.
    assert [s["verdict"] for s in steps] == [
        "allow",
        "allow",
        "block",
        "block",
        "allow",
        "allow",
        "block",
        "block",
    ]

    # Only escalate steps carry the inline hard-cell payload.
    escalated = [s for s in steps if s["mode"] == "escalate"]
    assert len(escalated) == 2
    assert all(s.get("hitl") and s["hitl"]["severity"] for s in escalated)

    # session_complete surfaces the priority-ordered HITL queue for the panel.
    assert [item["severity"] for item in complete["hitl_queue"]] == ["critical", "high"]
    assert all(item["reason"] and item["resolution"] for item in complete["hitl_queue"])
