from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from auditk.schema import Action, ActionType, Actor, FlowType, Step, Trace

from bossyk_sandbox.instruments.base import Decision, ProposedAction


def make_step(
    trace_id: str,
    proposed: ProposedAction,
    decision: Decision,
    *,
    step_id: str | None = None,
    timestamp: datetime | None = None,
) -> Step:
    """Map one GATE-scored tool call onto a spec-conformant auditk-spec Step."""
    return Step(
        step_id=step_id or f"step-{uuid4()}",
        trace_id=trace_id,
        timestamp=timestamp or datetime.now(UTC),
        actor=Actor.AGENT,
        declared_intent=proposed.declared_intent,
        action=Action(
            type=ActionType.TOOL_CALL,
            payload={
                "tool_name": proposed.tool_name,
                "arguments": proposed.arguments,
                "gate_verdict": decision.verdict.value,
                "gate_reason": decision.reason,
            },
        ),
        metadata={"gate_verdict": decision.verdict.value},
    )


def build_trace(
    trace_id: str,
    agent_config_ref: str,
    steps: list[Step],
    source_adapter: str = "bossyk-sandbox-gate@0.1",
) -> Trace:
    return Trace(
        trace_id=trace_id,
        flow_type=FlowType.GENERIC,
        agent_config_ref=agent_config_ref,
        steps=steps,
        source_adapter=source_adapter,
    )
