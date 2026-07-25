from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from auditk.schema import Action, ActionType, Actor, FlowType, Step, Trace

from bossyk_sandbox.compliance.attribution import CONTROLS_METADATA_KEY, controls_for_step
from bossyk_sandbox.instruments.base import Decision, ProposedAction, Verdict


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


def make_attested_step(
    trace_id: str,
    proposed: ProposedAction,
    *,
    auto_decision: Decision,
    final_verdict: Verdict,
    step_id: str | None = None,
    timestamp: datetime | None = None,
) -> Step:
    """Map one GATE-scored tool call onto a spec-conformant auditk-spec Step,
    attesting the FINAL verdict (possibly a manual override of the automatic
    decision) rather than the automatic decision alone.

    `overridden` is True whenever `final_verdict` differs from
    `auto_decision.verdict`. The step's payload/metadata always reflect the
    final verdict; `gate_reason` explains the automatic reasoning, prefixed
    with a manual-override note when the human overrode it.
    """
    overridden = final_verdict is not auto_decision.verdict
    if overridden:
        reason = (
            f"manual override of automatic {auto_decision.verdict.value}: {auto_decision.reason}"
        )
    else:
        reason = auto_decision.reason

    step = make_step(
        trace_id,
        proposed,
        Decision(final_verdict, reason),
        step_id=step_id,
        timestamp=timestamp,
    )
    step.metadata["automatic_verdict"] = auto_decision.verdict.value
    step.metadata["overridden"] = overridden
    step.metadata[CONTROLS_METADATA_KEY] = [
        tag.model_dump()
        for tag in controls_for_step(proposed, final_verdict, overridden=overridden)
    ]
    return step


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
