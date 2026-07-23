from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from bossyk_sandbox.console.artifacts import router as artifacts_router
from bossyk_sandbox.evidence.trace import build_trace, make_attested_step
from bossyk_sandbox.gate import Gate
from bossyk_sandbox.instruments.base import Verdict
from bossyk_sandbox.instruments.hardcoded_rule import RequireLookupBeforeCancel
from bossyk_sandbox.runtime.stub_agent import SCRIPTED_TOOL_CALLS

app = FastAPI(title="bossyk-sandbox console")
app.include_router(artifacts_router)

_connections: set[WebSocket] = set()


@dataclass
class HeldAction:
    """One gated proposal awaiting a manual decision, scoped to a single
    console session (Finding 6)."""

    action_id: str
    payload: dict[str, Any]
    event: asyncio.Event = field(default_factory=asyncio.Event)
    override: str | None = None


@dataclass
class ConsoleSession:
    """A single console run, scoped by `session_id` (Finding 6). Sessions
    are registered in `_sessions` synchronously (under `_sessions_lock`)
    before their task is scheduled, so a concurrent second `start_session`
    call sees it immediately rather than racing the task's first tick."""

    session_id: str
    held: HeldAction | None = None
    task: asyncio.Task[None] | None = None


# Process-global session registry: at most one entry at a time (a second
# `start_session` while one is running reports "already_running"). Scoped by
# session_id/action_id (Finding 6) rather than the single implicit slot the
# old `_held`/`_decision_event`/`_decision_value`/`_session_running` globals
# provided.
_sessions: dict[str, ConsoleSession] = {}
_sessions_lock = asyncio.Lock()


async def _broadcast(event: dict[str, Any]) -> None:
    dead = set()
    for ws in _connections:
        try:
            await ws.send_json(event)
        except Exception:
            dead.add(ws)
    _connections.difference_update(dead)


async def _run_stub_session(session: ConsoleSession, hold_timeout_s: float = 5.0) -> None:
    """Drive the deterministic stub tool-call sequence through the real Gate,
    holding each proposed action for `hold_timeout_s` so a client can submit
    a manual override (identified by `session.session_id`/`action_id`)
    before the auto verdict is committed."""
    gate = Gate(instruments=[RequireLookupBeforeCancel()])
    steps = []

    try:
        for i, call in enumerate(SCRIPTED_TOOL_CALLS):
            auto_decision = gate.score(call)
            payload = {
                "tool_name": call.tool_name,
                "arguments": call.arguments,
                "auto_verdict": auto_decision.verdict.value,
                "auto_reason": auto_decision.reason,
            }
            held = HeldAction(action_id=f"{session.session_id}-action-{i}", payload=payload)
            session.held = held
            await _broadcast(
                {
                    "type": "held",
                    "session_id": session.session_id,
                    "action_id": held.action_id,
                    **payload,
                }
            )

            try:
                await asyncio.wait_for(held.event.wait(), timeout=hold_timeout_s)
            except TimeoutError:
                pass

            final_verdict = Verdict(held.override) if held.override else auto_decision.verdict
            if final_verdict is Verdict.ALLOW:
                gate.record(call)
            session.held = None

            step = make_attested_step(
                trace_id="console-session-1",
                proposed=call,
                auto_decision=auto_decision,
                final_verdict=final_verdict,
            )
            steps.append(step)
            await _broadcast(
                {
                    "type": "step",
                    "tool_name": call.tool_name,
                    "arguments": call.arguments,
                    "verdict": final_verdict.value,
                    "overridden": final_verdict is not auto_decision.verdict,
                }
            )

        build_trace(trace_id="console-session-1", agent_config_ref="console-stub@0.1", steps=steps)
        await _broadcast({"type": "session_complete", "step_count": len(steps)})
    finally:
        _sessions.pop(session.session_id, None)


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(Path(__file__).parent / "index.html")


@app.post("/session/run")
async def start_session(hold_timeout_s: float = 5.0) -> dict[str, str]:
    async with _sessions_lock:
        if _sessions:
            return {"status": "already_running"}
        session_id = uuid4().hex
        session = ConsoleSession(session_id=session_id)
        _sessions[session_id] = session
        session.task = asyncio.create_task(_run_stub_session(session, hold_timeout_s))
    return {"status": "started", "session_id": session_id}


@app.post("/session/decide")
async def decide(body: dict[str, str]) -> dict[str, str]:
    verdict = body.get("verdict")
    if verdict not in (Verdict.ALLOW.value, Verdict.BLOCK.value):
        return {"status": "invalid_verdict"}

    session = _sessions.get(body.get("session_id", ""))
    if session is None:
        return {"status": "unknown_session"}

    held = session.held
    if held is None:
        return {"status": "no_action_held"}

    if body.get("action_id") != held.action_id:
        return {"status": "stale_action"}

    held.override = verdict
    held.event.set()
    return {"status": "accepted"}


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    _connections.add(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        _connections.discard(websocket)
