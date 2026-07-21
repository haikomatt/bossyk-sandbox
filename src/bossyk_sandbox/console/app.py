from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from bossyk_sandbox.evidence.trace import build_trace, make_step
from bossyk_sandbox.gate import Gate
from bossyk_sandbox.instruments.base import Decision, Verdict
from bossyk_sandbox.instruments.hardcoded_rule import RequireLookupBeforeCancel
from bossyk_sandbox.runtime.stub_agent import SCRIPTED_TOOL_CALLS

app = FastAPI(title="bossyk-sandbox console")

_connections: set[WebSocket] = set()
_held: dict[str, Any] | None = None
_decision_event = asyncio.Event()
_decision_value: str | None = None
_session_running = False


async def _broadcast(event: dict[str, Any]) -> None:
    dead = set()
    for ws in _connections:
        try:
            await ws.send_json(event)
        except Exception:
            dead.add(ws)
    _connections.difference_update(dead)


async def _run_stub_session(hold_timeout_s: float = 5.0) -> None:
    """Drive the deterministic stub tool-call sequence through the real Gate,
    holding each proposed action for `hold_timeout_s` so a client can submit
    a manual override before the auto verdict is committed."""
    global _held, _decision_value, _session_running
    _session_running = True
    gate = Gate(instruments=[RequireLookupBeforeCancel()])
    steps = []

    try:
        for call in SCRIPTED_TOOL_CALLS:
            auto_decision = gate.score(call)
            _held = {
                "tool_name": call.tool_name,
                "arguments": call.arguments,
                "auto_verdict": auto_decision.verdict.value,
                "auto_reason": auto_decision.reason,
            }
            _decision_value = None
            _decision_event.clear()
            await _broadcast({"type": "held", **_held})

            try:
                await asyncio.wait_for(_decision_event.wait(), timeout=hold_timeout_s)
            except TimeoutError:
                pass

            final_verdict = Verdict(_decision_value) if _decision_value else auto_decision.verdict
            gate.record(call)
            _held = None

            step = make_step(
                trace_id="console-session-1",
                proposed=call,
                decision=Decision(final_verdict, auto_decision.reason),
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
        _session_running = False


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(Path(__file__).parent / "index.html")


@app.post("/session/run")
async def start_session() -> dict[str, str]:
    if _session_running:
        return {"status": "already_running"}
    asyncio.create_task(_run_stub_session())
    return {"status": "started"}


@app.post("/session/decide")
async def decide(body: dict[str, str]) -> dict[str, str]:
    global _decision_value
    verdict = body.get("verdict")
    if verdict not in (Verdict.ALLOW.value, Verdict.BLOCK.value):
        return {"status": "invalid_verdict"}
    if _held is None:
        return {"status": "no_action_held"}
    _decision_value = verdict
    _decision_event.set()
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
