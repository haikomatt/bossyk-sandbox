from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

from auditk.schema import Step
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from tau2.domains.retail.environment import get_environment as get_retail_environment

from bossyk_sandbox.compliance.attribution import CONTROLS_METADATA_KEY
from bossyk_sandbox.compliance.frameworks import load_frameworks
from bossyk_sandbox.console.artifacts import router as artifacts_router
from bossyk_sandbox.console.live import (
    advance_to_interrupt,
    authority_for,
    hitl_item,
    initial_input,
    order_reader_from_toolkit,
    resolve_call,
    resume_after,
    sort_hitl_queue,
    thread_config,
)
from bossyk_sandbox.console.replay import (
    ReplayPreset,
    build_gate,
    build_hitl_queue,
    load_replay_preset,
    trace_id_for,
)
from bossyk_sandbox.evidence.trace import build_trace, make_attested_step
from bossyk_sandbox.gate import Gate
from bossyk_sandbox.instruments.base import ProposedAction, Verdict
from bossyk_sandbox.instruments.hardcoded_rule import RequireLookupBeforeCancel
from bossyk_sandbox.runtime.langgraph_agent import (
    AgentSession,
    build_weakened_retail_agent_session,
)
from bossyk_sandbox.runtime.stub_agent import SCRIPTED_TOOL_CALLS
from bossyk_sandbox.standing import TimedAction, retail_standing_grants
from bossyk_sandbox.standing_amount import OrderReader, resolve_amount

app = FastAPI(title="bossyk-sandbox console")
app.include_router(artifacts_router)

# Resolve control refs to human framework/control names once, so each
# broadcast step can carry displayable compliance tags without the console
# page having to fetch and index the catalogue itself.
_CONTROL_LABELS: dict[str, dict[str, str]] = {
    f"{framework.id}:{control.id}": {
        "framework": framework.name,
        "control": control.ref,
        "title": control.title,
    }
    for framework in load_frameworks().entries
    for control in framework.controls
}


def _display_controls(step: Step) -> list[dict[str, str]]:
    """The step's compliance control tags, each resolved to its human
    framework/control names alongside its ref and basis. An unresolved ref
    (should not happen -- the tagger only emits catalogue refs) is dropped."""
    display: list[dict[str, str]] = []
    for tag in step.metadata.get(CONTROLS_METADATA_KEY, []):
        label = _CONTROL_LABELS.get(tag["ref"])
        if label is None:
            continue
        display.append({"ref": tag["ref"], "basis": tag["basis"], **label})
    return display


# F1 evidence-browser SPA (frontend/). Build output is not committed, so
# this mount is conditional: absent a build, `/app` simply 404s and every
# other route (including all existing tests) is unaffected.
_frontend_dist = Path(__file__).resolve().parent.parent.parent.parent / "frontend" / "dist"
if _frontend_dist.is_dir():
    app.mount("/app", StaticFiles(directory=_frontend_dist, html=True), name="frontend")

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
                    "controls": _display_controls(step),
                }
            )

        build_trace(trace_id="console-session-1", agent_config_ref="console-stub@0.1", steps=steps)
        await _broadcast({"type": "session_complete", "step_count": len(steps)})
    finally:
        _sessions.pop(session.session_id, None)


async def _run_replay_session(
    session: ConsoleSession, preset: ReplayPreset, pacing_s: float = 1.0
) -> None:
    """Non-interactive playback of a committed replay preset. Each turn is
    briefly surfaced as a "held" event (so the UI can render the hold beat)
    then advanced automatically after `pacing_s` -- there is no decision to
    make, this replays a fixed, artifact-grounded gate-save. Reuses the same
    real domain gate + attested-step + broadcast plumbing as the live console,
    so the event shape is identical to `_run_stub_session`'s."""
    gate = build_gate(preset)
    steps = []
    harm_prevented = 0

    try:
        for i, turn in enumerate(preset.turns):
            decision = gate.score(turn.proposed)
            action_id = f"{session.session_id}-action-{i}"
            await _broadcast(
                {
                    "type": "held",
                    "replay": True,
                    "preset": preset.preset_id,
                    "session_id": session.session_id,
                    "action_id": action_id,
                    "tool_name": turn.proposed.tool_name,
                    "arguments": turn.proposed.arguments,
                    "auto_verdict": decision.verdict.value,
                    "auto_reason": decision.reason,
                    "label": turn.label,
                    "role": turn.role,
                    "probe_id": turn.probe_id,
                    "mode": turn.mode,
                }
            )
            await asyncio.sleep(pacing_s)

            if decision.verdict is Verdict.ALLOW:
                gate.record(turn.proposed)
            step = make_attested_step(
                trace_id=trace_id_for(preset),
                proposed=turn.proposed,
                auto_decision=decision,
                final_verdict=decision.verdict,
            )
            steps.append(step)
            if turn.role == "crossing" and decision.verdict is Verdict.BLOCK:
                harm_prevented += 1

            step_event = {
                "type": "step",
                "tool_name": turn.proposed.tool_name,
                "arguments": turn.proposed.arguments,
                "verdict": decision.verdict.value,
                "overridden": False,
                "controls": _display_controls(step),
                "label": turn.label,
                "role": turn.role,
                "probe_id": turn.probe_id,
                "mode": turn.mode,
                "mode_reason": turn.mode_reason,
            }
            if turn.mode == "escalate" and turn.hitl is not None:
                step_event["hitl"] = turn.hitl
            await _broadcast(step_event)

        build_trace(
            trace_id=trace_id_for(preset),
            agent_config_ref=f"replay:{preset.preset_id}",
            steps=steps,
        )
        await _broadcast(
            {
                "type": "session_complete",
                "step_count": len(steps),
                "harm_prevented": harm_prevented,
                "harm_delta": harm_prevented,
                "preset": preset.preset_id,
                "source_artifact": preset.source_artifact,
                "story_claim": preset.story_claim,
                "hitl_queue": build_hitl_queue(preset),
            }
        )
    finally:
        _sessions.pop(session.session_id, None)


async def _run_live_session(
    session: ConsoleSession,
    agent_session: AgentSession,
    reader: OrderReader | None = None,
    pacing_s: float = 1.0,
) -> None:
    """Non-interactive playback of a LIVE agent session: drive the real
    LangGraph graph (offloading each blocking LLM segment to a thread), derive
    the resolution mode per held call from the gate's own verdict, and broadcast
    the same event shape as the preset replay. Enforces the gate — a held call is
    resumed with its automatic verdict, never a human override — so no blocked
    tool executes. BILLABLE: `agent_session` runs a real LLM per turn.

    `reader` (an `OrderReader` adapted from the live tau2 environment via
    `order_reader_from_toolkit`) resolves each held call's amount for §F's
    amount-gated grants; `None` (the default, and every call site before P3)
    leaves amounts unresolved, which only matters once a grant sets
    `max_amount` -- `retail_standing_grants()` stays count-only (decision #4),
    so this is a no-op today, wired ready for an opt-in amount-gated policy."""
    graph = agent_session.graph
    config = thread_config(session.session_id)
    hitl_queue: list[dict[str, str]] = []
    grants = retail_standing_grants()  # §F: committed per-domain standing policy
    history: list[TimedAction] = []  # allowed actions consume standing authority

    try:
        payload = await asyncio.to_thread(advance_to_interrupt, graph, initial_input(), config)
        i = 0
        while payload is not None:
            held = payload
            now = time.time()  # stamps allowed actions so windowed grants can expire
            proposed = ProposedAction(
                str(held["tool_name"]), cast(dict[str, Any], held["arguments"])
            )
            amount = resolve_amount(proposed, reader) if reader is not None else None
            verdict, decision = resolve_call(
                held, authority=authority_for(held, history, grants, now=now, amount=amount)
            )
            if verdict is Verdict.ALLOW:
                history.append(TimedAction(proposed, now, amount=amount))
            await _broadcast(
                {
                    "type": "held",
                    "live": True,
                    "session_id": session.session_id,
                    "action_id": f"{session.session_id}-action-{i}",
                    "tool_name": held["tool_name"],
                    "arguments": held["arguments"],
                    "auto_verdict": verdict.value,
                    "auto_reason": held.get("auto_reason"),
                    "label": held["tool_name"],
                    "mode": decision.mode,
                }
            )
            await asyncio.sleep(pacing_s)

            # Advancing runs execute_node for THIS call (appending its attested
            # step) then pauses at the next held call (or END).
            payload = await asyncio.to_thread(
                advance_to_interrupt, graph, resume_after(held), config
            )
            step = agent_session.steps[i]
            step_event: dict[str, Any] = {
                "type": "step",
                "tool_name": held["tool_name"],
                "arguments": held["arguments"],
                "verdict": verdict.value,
                "overridden": False,
                "controls": _display_controls(step),
                "label": held["tool_name"],
                "mode": decision.mode,
                "mode_reason": decision.reason,
            }
            if decision.mode == "escalate" and decision.hitl is not None:
                hitl_queue.append(hitl_item(held, decision))
                step_event["hitl"] = decision.hitl
            await _broadcast(step_event)
            i += 1

        build_trace(
            trace_id=agent_session.trace_id,
            agent_config_ref="live:weakened-retail",
            steps=agent_session.steps,
        )
        await _broadcast(
            {
                "type": "session_complete",
                "live": True,
                "step_count": len(agent_session.steps),
                "hitl_queue": sort_hitl_queue(hitl_queue),
            }
        )
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


@app.post("/session/replay")
async def start_replay_session(preset_id: str, pacing_s: float = 1.0) -> dict[str, str]:
    """Start a non-interactive preset replay. Shares the single-session slot
    with the interactive stub session (at most one console run at a time)."""
    async with _sessions_lock:
        if _sessions:
            return {"status": "already_running"}
        try:
            preset = load_replay_preset(preset_id)
        except FileNotFoundError:
            return {"status": "unknown_preset"}
        session_id = uuid4().hex
        session = ConsoleSession(session_id=session_id)
        _sessions[session_id] = session
        session.task = asyncio.create_task(_run_replay_session(session, preset, pacing_s))
    return {"status": "started", "session_id": session_id}


@app.post("/session/live")
async def start_live_session(pacing_s: float = 1.0) -> dict[str, str]:
    """Start a LIVE weakened-retail agent session (modes derived from the real
    gate + agent, not a preset). BILLABLE: running the agent calls the LLM per
    turn.

    Hard-gated behind `RUN_LIVE_CONSOLE=1`: a key being present in the
    environment is never sufficient on its own. Without the flag this returns
    "live_disabled" BEFORE building the session or resolving any key, so a stray
    click or POST can never bill. With the flag but no key it returns
    "no_api_key" (the graph is constructed but never run). Shares the
    single-session slot.

    Resolves the real tau2 retail environment itself (rather than letting
    `build_weakened_retail_agent_session` resolve one internally) so the SAME
    environment/toolkit backs both the agent graph and the §F amount-oracle
    reader (`order_reader_from_toolkit`) -- one object, one source of truth
    for order state. Non-billable: no LLM call, just the local retail DB."""
    if os.environ.get("RUN_LIVE_CONSOLE") != "1":
        return {"status": "live_disabled"}

    async with _sessions_lock:
        if _sessions:
            return {"status": "already_running"}
        try:
            environment = await asyncio.to_thread(get_retail_environment)
            agent_session = await asyncio.to_thread(
                build_weakened_retail_agent_session, environment=environment
            )
        except RuntimeError:
            return {"status": "no_api_key"}
        reader = order_reader_from_toolkit(environment.tools)
        session_id = uuid4().hex
        session = ConsoleSession(session_id=session_id)
        _sessions[session_id] = session
        session.task = asyncio.create_task(
            _run_live_session(session, agent_session, reader, pacing_s)
        )
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
