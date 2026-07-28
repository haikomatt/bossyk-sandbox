"""Drive a live LangGraph agent session through the gate, deriving a resolution
mode per proposed tool call.

`run_live_session` is the pure, deterministic core (the analogue of
`replay.drive_replay`): it drives the real agent graph's interrupt chain,
resumes each held call with the gate's OWN verdict (enforcing the gate -- no
human override, so a blocked write never executes), derives the resolution mode
from that verdict + the tool's consequence, and accumulates the hard-cell HITL
queue.

It is billable ONLY when the injected agent session was built with a real LLM
client; built with a fake LLM + fake tau2 env it runs entirely offline (see
tests/unit/test_live_session.py). This module never constructs an LLM itself.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, cast

from auditk.schema import Step
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from bossyk_sandbox.console.modes import SEVERITY_RANK, derive_mode
from bossyk_sandbox.instruments.base import Verdict
from bossyk_sandbox.runtime.langgraph_agent import AgentSession, AgentState

_DEFAULT_USER_MESSAGE = "Hi, I need help with a couple of orders on my account."


@dataclass(frozen=True)
class LiveResult:
    """The outcome of driving a live session: per-call gate verdicts + derived
    resolution modes, the attested steps the graph produced, and the
    priority-ordered hard-cell HITL queue."""

    verdicts: list[str]
    modes: list[str]
    steps: list[Step]
    hitl_queue: list[dict[str, str]]


def _interrupt_payload(chunk: Any) -> dict[str, object] | None:
    """The approve-node interrupt payload in a stream chunk, or None. LangGraph
    surfaces an interrupt as `{"__interrupt__": (Interrupt(value=...),)}`."""
    interrupts = chunk.get("__interrupt__")
    if not interrupts:
        return None
    payload = interrupts[0].value
    assert isinstance(payload, dict)
    return payload


def run_live_session(
    session: AgentSession,
    *,
    thread_id: str = "live-session-1",
    user_message: str = _DEFAULT_USER_MESSAGE,
) -> LiveResult:
    graph = session.graph
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

    verdicts: list[str] = []
    modes: list[str] = []
    hitl_queue: list[dict[str, str]] = []

    initial = cast(AgentState, {"messages": [HumanMessage(content=user_message)]})
    stream: Iterator[Any] = graph.stream(initial, config)
    while True:
        payload: dict[str, object] | None = None
        for chunk in stream:
            found = _interrupt_payload(chunk)
            if found is not None:
                payload = found
        if payload is None:
            break  # the graph reached END with no further held call

        verdict = Verdict(str(payload["auto_verdict"]))
        decision = derive_mode(str(payload["tool_name"]), verdict)
        verdicts.append(verdict.value)
        modes.append(decision.mode)
        if decision.mode == "escalate" and decision.hitl is not None:
            hitl_queue.append(
                {
                    **decision.hitl,
                    "label": str(payload["tool_name"]),
                    "tool_name": str(payload["tool_name"]),
                }
            )

        # Resume enforcing the gate's own verdict (no human override).
        stream = graph.stream(Command(resume=payload["auto_verdict"]), config)

    hitl_queue.sort(key=lambda item: SEVERITY_RANK.get(item["severity"], 99))
    return LiveResult(
        verdicts=verdicts,
        modes=modes,
        steps=list(session.steps),
        hitl_queue=hitl_queue,
    )
