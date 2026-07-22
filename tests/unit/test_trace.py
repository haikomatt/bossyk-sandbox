from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema

from bossyk_sandbox.evidence.trace import build_trace, make_attested_step, make_step
from bossyk_sandbox.gate import Gate
from bossyk_sandbox.instruments.base import Decision, ProposedAction, Verdict
from bossyk_sandbox.instruments.hardcoded_rule import RequireLookupBeforeCancel

SPEC_DIR = (Path(__file__).parents[2] / ".." / "auditk-spec" / "spec" / "v0.1").resolve()


def _load_schema(name: str) -> dict[str, Any]:
    schema: dict[str, Any] = json.loads((SPEC_DIR / name).read_text())
    return schema


def test_spec_schemas_are_reachable() -> None:
    assert (SPEC_DIR / "trace.schema.json").exists()


def test_step_and_trace_are_spec_conformant() -> None:
    gate = Gate(instruments=[RequireLookupBeforeCancel()])
    proposed = ProposedAction("get_reservation_details", {"reservation_id": "R1"})
    decision = gate.score(proposed)
    step = make_step(trace_id="t-1", proposed=proposed, decision=decision)
    trace = build_trace(trace_id="t-1", agent_config_ref="cfg-1", steps=[step])

    jsonschema.validate(
        instance=trace.model_dump(mode="json"), schema=_load_schema("trace.schema.json")
    )


def test_blocked_step_records_block_verdict_in_action_payload() -> None:
    gate = Gate(instruments=[RequireLookupBeforeCancel()])
    proposed = ProposedAction("cancel_reservation", {"reservation_id": "R1"})
    decision = gate.score(proposed)
    step = make_step(trace_id="t-1", proposed=proposed, decision=decision)

    assert step.action.payload["gate_verdict"] == "block"


def test_make_step_persists_declared_intent() -> None:
    gate = Gate(instruments=[RequireLookupBeforeCancel()])
    proposed = ProposedAction(
        "get_reservation_details",
        {"reservation_id": "R1"},
        declared_intent="look up R1 before cancelling",
    )
    decision = gate.score(proposed)
    step = make_step(trace_id="t-1", proposed=proposed, decision=decision)

    assert step.declared_intent == "look up R1 before cancelling"


def test_make_step_declared_intent_defaults_to_none() -> None:
    gate = Gate(instruments=[RequireLookupBeforeCancel()])
    proposed = ProposedAction("get_reservation_details", {"reservation_id": "R1"})
    decision = gate.score(proposed)
    step = make_step(trace_id="t-1", proposed=proposed, decision=decision)

    assert step.declared_intent is None


def test_make_attested_step_with_no_override_reflects_the_automatic_allow() -> None:
    proposed = ProposedAction("get_reservation_details", {"reservation_id": "R1"})
    auto_decision = Decision(Verdict.ALLOW, "no instrument blocked")

    step = make_attested_step(
        trace_id="t-1",
        proposed=proposed,
        auto_decision=auto_decision,
        final_verdict=Verdict.ALLOW,
    )

    assert step.action.payload["gate_verdict"] == "allow"
    assert step.action.payload["gate_reason"] == "no instrument blocked"
    assert step.metadata["gate_verdict"] == "allow"
    assert step.metadata["automatic_verdict"] == "allow"
    assert step.metadata["overridden"] is False


def test_make_attested_step_manual_allow_over_automatic_block_attests_the_allow() -> None:
    proposed = ProposedAction("cancel_reservation", {"reservation_id": "R9"})
    auto_decision = Decision(
        Verdict.BLOCK, "cancel_reservation for R9 has no prior lookup in this session"
    )

    step = make_attested_step(
        trace_id="t-1",
        proposed=proposed,
        auto_decision=auto_decision,
        final_verdict=Verdict.ALLOW,
    )

    assert step.action.payload["gate_verdict"] == "allow"
    assert step.metadata["gate_verdict"] == "allow"
    assert step.metadata["automatic_verdict"] == "block"
    assert step.metadata["overridden"] is True
    assert auto_decision.reason in step.action.payload["gate_reason"]
    assert "manual override of automatic block" in step.action.payload["gate_reason"]


def test_make_attested_step_manual_block_over_automatic_allow_attests_the_block() -> None:
    proposed = ProposedAction("get_reservation_details", {"reservation_id": "R1"})
    auto_decision = Decision(Verdict.ALLOW, "no instrument blocked")

    step = make_attested_step(
        trace_id="t-1",
        proposed=proposed,
        auto_decision=auto_decision,
        final_verdict=Verdict.BLOCK,
    )

    assert step.action.payload["gate_verdict"] == "block"
    assert step.metadata["gate_verdict"] == "block"
    assert step.metadata["automatic_verdict"] == "allow"
    assert step.metadata["overridden"] is True
    assert auto_decision.reason in step.action.payload["gate_reason"]
    assert "manual override of automatic allow" in step.action.payload["gate_reason"]
