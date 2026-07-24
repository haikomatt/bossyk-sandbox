from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

from auditk.adapters.protocols import Stimulus  # noqa: F401 -- see model_rebuild() below
from auditk.schema import ProbeDefinition, Step, Trace
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from bossyk_sandbox.conditions.live_boundary import parse_family
from bossyk_sandbox.evidence.trace import build_trace, make_step
from bossyk_sandbox.instruments.base import (
    Decision,
    InstrumentVerdict,
    ProposedAction,
    SlowInstrument,
    Verdict,
)
from bossyk_sandbox.runtime.langgraph_agent import (
    AgentSession,
    build_airline_agent_session,
    build_retail_agent_session,
    build_weakened_retail_agent_session,
)
from bossyk_sandbox.scenarios.runner import VERDICT_METADATA_KEY
from bossyk_sandbox.scoring.latency import Clock, LatencyRecord, timed

# ProbeDefinition.stimulus is typed via a TYPE_CHECKING-only import of
# Stimulus in auditk.schema, so the forward ref must be resolved here (after
# importing Stimulus above) before any ProbeDefinition is constructed or
# model_validate'd -- mirrors conditions.retention's pattern. Idempotent (a
# no-op if some other import path already triggered it in this process).
ProbeDefinition.model_rebuild()

# This module is the "live" injection driver named in
# phase-live-h2h4-on-crossings.md: take a frozen crossing (ProbeDefinition,
# stimulus.payload["text"]) and drive it through a live agent session,
# capturing what the agent PROPOSED (session.steps, blocked calls included)
# and what actually EXECUTED (gate.history) for the boundary oracle
# (conditions.live_boundary) and post-hoc policy scoring to consume.
#
# `run_live_airline_session` / `run_live_retail_session` are the ONLY
# network-touching, billable functions in this module (real Fireworks
# agent calls via build_{airline,retail}_agent_session). They must never be
# invoked by the deterministic unit/e2e suite -- only
# scripts/live_h2h4_bench.py calls them, gated behind RUN_LIVE_H2_E2E=1.
# Every test in this module's own test file drives `replay_crossing` and
# `score_policy_post_hoc` with a FAKE `run_session` / `SlowInstrument`.


class LiveSessionResult(Protocol):
    """What `replay_crossing` needs back from a live (or fake) agent run:
    every call the agent PROPOSED (gate-scored, blocked calls included) and
    every call that actually EXECUTED. Satisfied structurally by
    `LiveRunResult` (the real runners' return value) or any fake with these
    two attributes -- deterministic tests never need to import
    `LiveRunResult` itself.

    Declared as read-only properties (not plain attributes) so a frozen
    dataclass like `LiveRunResult` satisfies it structurally under mypy --
    a Protocol with plain mutable-attribute members requires a *settable*
    variable, which a frozen dataclass's read-only attribute can never be.
    """

    @property
    def proposed(self) -> list[ProposedAction]: ...

    @property
    def executed(self) -> list[ProposedAction]: ...


@dataclass(frozen=True)
class LiveRunResult:
    """Concrete `LiveSessionResult`: what `run_live_airline_session` /
    `run_live_retail_session` return, and what a deterministic test's fake
    `run_session` can return too. `trace` is the auditk Trace built from
    the session's proposed Steps (gate_verdict included per step) -- kept
    alongside the bare `proposed`/`executed` lists because those lose the
    per-call gate_verdict that only the Trace's Step payloads carry."""

    proposed: list[ProposedAction]
    executed: list[ProposedAction]
    trace: Trace | None = None
    # §15B+: wall-clock of each tau2 tool call the session actually executed
    # (empty when every proposed action was gate-blocked pre-execution).
    tool_latency: list[LatencyRecord] = field(default_factory=list)


@dataclass(frozen=True)
class CrossingReplay:
    """One frozen crossing (`ProbeDefinition`), replayed live: the probe
    itself, its domain + boundary (parsed from `probe.family`, e.g.
    `"airline-cancel_without_lookup"` -> `("airline", "cancel_without_lookup")`),
    and what the live session proposed/executed."""

    probe: ProbeDefinition
    domain: str
    boundary: str
    proposed: list[ProposedAction]
    executed: list[ProposedAction]
    trace: Trace | None = None
    tool_latency: list[LatencyRecord] = field(default_factory=list)


def replay_crossing(
    probe: ProbeDefinition, run_session: Callable[[str], LiveSessionResult]
) -> CrossingReplay:
    """Extracts `probe.stimulus.payload["text"]` (the injection payload;
    all 56 frozen crossings are `channel="user"`, so direct replay as the
    single user turn preserves channel fidelity -- see the Scout's
    "Confirmed ready" section) and drives it through `run_session`.

    Raises `ValueError` if `probe.family` has no recognised domain prefix
    (`"airline-"` / `"retail-"`) -- a frozen crossing without one is
    malformed, not a bare-boundary case (bare-boundary defaulting is
    `conditions.live_boundary.boundary_spec_for`'s concern, for callers
    that already know the domain out of band; a real corpus probe always
    carries it in `family`).
    """
    domain, boundary = parse_family(probe.family)
    if domain is None:
        raise ValueError(
            f"probe {probe.probe_id!r} has family {probe.family!r} with no recognised "
            "domain prefix -- expected '<domain>-<boundary>', e.g. "
            "'airline-cancel_without_lookup'"
        )
    payload = probe.stimulus.payload["text"]
    result = run_session(payload)
    return CrossingReplay(
        probe=probe,
        domain=domain,
        boundary=boundary,
        proposed=list(result.proposed),
        executed=list(result.executed),
        trace=getattr(result, "trace", None),
        tool_latency=list(getattr(result, "tool_latency", [])),
    )


def _drive_session(session: AgentSession, payload: str, trace_id: str) -> None:
    """Drives `session.graph` to completion, resuming every interrupt with
    its own automatic verdict -- mirrors scripts/live_demo.py's and
    tests/unit/test_langgraph_agent.py's drive loop. No manual overrides:
    the L1 live run has no human in the loop."""
    config = {"configurable": {"thread_id": trace_id}}
    result = session.graph.invoke(  # type: ignore[call-overload]
        {"messages": [HumanMessage(content=payload)]}, config=config
    )
    while "__interrupt__" in result:
        interrupt_payload = result["__interrupt__"][0].value
        result = session.graph.invoke(  # type: ignore[call-overload]
            Command(resume=interrupt_payload["auto_verdict"]), config=config
        )


def _live_run_result(session: AgentSession, trace_id: str, agent_config_ref: str) -> LiveRunResult:
    proposed = [
        ProposedAction(step.action.payload["tool_name"], step.action.payload["arguments"])
        for step in session.steps
    ]
    assert session.gate is not None
    executed = session.gate.history
    trace = build_trace(trace_id=trace_id, agent_config_ref=agent_config_ref, steps=session.steps)
    return LiveRunResult(
        proposed=proposed, executed=executed, trace=trace, tool_latency=list(session.tool_latency)
    )


def run_live_airline_session(payload: str) -> LiveRunResult:
    """REAL, network-touching, billable runner: builds a live airline agent
    session (Fireworks kimi-k2p6, requires FIREWORKS_API_KEY), drives it
    against `payload` as the single user turn, and returns its
    proposed/executed history. Never call this from the deterministic
    suite -- only scripts/live_h2h4_bench.py does, gated behind
    RUN_LIVE_H2_E2E=1."""
    trace_id = f"live-h2-airline-{uuid.uuid4()}"
    session = build_airline_agent_session(trace_id=trace_id)
    _drive_session(session, payload, trace_id)
    return _live_run_result(session, trace_id, agent_config_ref="live-h2h4-airline@0.1")


def run_live_retail_session(payload: str) -> LiveRunResult:
    """Retail counterpart of `run_live_airline_session`. Same caveats:
    real, network-touching, billable -- never called by the deterministic
    suite."""
    trace_id = f"live-h2-retail-{uuid.uuid4()}"
    session = build_retail_agent_session(trace_id=trace_id)
    _drive_session(session, payload, trace_id)
    return _live_run_result(session, trace_id, agent_config_ref="live-h2h4-retail@0.1")


def run_live_weakened_retail_session(payload: str) -> LiveRunResult:
    """dir 1: like run_live_retail_session but against the UNDER-SPECIFIED
    (weakened-policy) retail agent -- real, network-touching, billable. Never
    called by the deterministic suite -- only scripts/live_h2h4_bench.py does,
    gated behind RUN_LIVE_H2_E2E=1."""
    trace_id = f"live-h2-retail-weak-{uuid.uuid4()}"
    session = build_weakened_retail_agent_session(trace_id=trace_id)
    _drive_session(session, payload, trace_id)
    return _live_run_result(session, trace_id, agent_config_ref="live-h2h4-retail-weak@0.1")


@dataclass(frozen=True)
class PolicyScoredCrossing:
    """One crossing's post-hoc policy annotation: the replay itself, a
    spec-conformant Trace built from its PROPOSED calls (evidentiary
    record, mirroring scenarios.runner.run_scenario's pattern), the raw
    per-step InstrumentVerdicts (order-aligned with `crossing.proposed`),
    and the wall-clock LatencyRecord for each `.annotate()` call."""

    crossing: CrossingReplay
    trace: Trace
    verdicts: list[InstrumentVerdict]
    latency: list[LatencyRecord] = field(default_factory=list)


def score_policy_post_hoc(
    crossing: CrossingReplay,
    policy: SlowInstrument,
    *,
    clock: Clock = time.perf_counter,
) -> PolicyScoredCrossing:
    """Runs `policy` (e.g. `instruments.policy.PolicyInstrument`, or a fake
    `SlowInstrument` in tests) over every call the agent PROPOSED in
    `crossing` -- blocked calls included, since the policy judge scores
    what the agent *tried*, not just what executed -- in order, each
    against the calls proposed *before* it as `history`. This mirrors
    `scenarios.runner.run_scenario`'s per-step slow-instrument pattern
    without re-deciding a gate verdict (that already happened live, or
    didn't happen at all for a non-structural boundary): the rebuilt Trace
    attests a fixed `ALLOW` placeholder, documented as such, never the
    live verdict.

    Token-usage accounting is intentionally NOT this function's concern:
    if `policy` is a `PolicyInstrument` with `on_call` wired to a list
    (`scoring.cost.JudgeCallRecord` accumulator), usage capture happens as
    a side effect of `policy.annotate()` itself, transparent to this
    driver.
    """
    steps: list[Step] = []
    verdicts: list[InstrumentVerdict] = []
    latency: list[LatencyRecord] = []
    for index, proposed in enumerate(crossing.proposed):
        history = crossing.proposed[:index]

        def _score(
            proposed: ProposedAction = proposed, history: list[ProposedAction] = history
        ) -> InstrumentVerdict:
            return policy.annotate(proposed, history)

        verdict, record = timed(policy.name, _score, clock=clock)
        verdicts.append(verdict)
        latency.append(record)

        step = make_step(
            trace_id=crossing.probe.probe_id,
            proposed=proposed,
            decision=Decision(
                Verdict.ALLOW, "post-hoc policy replay: gate verdict already decided live"
            ),
            step_id=f"{crossing.probe.probe_id}-posthoc-{index}",
        )
        step.metadata[VERDICT_METADATA_KEY] = {
            policy.name: {"label": verdict.label, "detail": verdict.detail}
        }
        steps.append(step)

    trace = build_trace(
        trace_id=crossing.probe.probe_id,
        agent_config_ref="live-h2h4-posthoc@0.1",
        steps=steps,
    )
    return PolicyScoredCrossing(crossing=crossing, trace=trace, verdicts=verdicts, latency=latency)
