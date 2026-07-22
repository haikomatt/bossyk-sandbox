from __future__ import annotations

from dataclasses import dataclass

from auditk.schema import Step, Trace

from bossyk_sandbox.evidence.trace import build_trace, make_step
from bossyk_sandbox.gate import Gate, TwoSpeedGate
from bossyk_sandbox.instruments.base import (
    Decision,
    Instrument,
    InstrumentVerdict,
    SlowInstrument,
    Verdict,
)
from bossyk_sandbox.instruments.hardcoded_rule import RequireLookupBeforeCancel
from bossyk_sandbox.instruments.outcome_key import OutcomeKeyLookup
from bossyk_sandbox.scenarios.loader import Scenario
from bossyk_sandbox.scoring.confusion import GateOutcomeRecord
from bossyk_sandbox.scoring.orthogonality import StepMembership

VERDICT_METADATA_KEY = "bossyk_sandbox_verdict"

# Shared with auditk's TaxonomyLabel / bossyk's DRIFT_LABELS vocabulary
# (see SCOUT.md Phase 1 #2) — labels outside this set (faithful,
# benign_elaboration, neutral) are non-firing.
FIRING_LABELS = {"goal_deviation", "instruction_noncompliance", "undeclared_goal"}


def default_fast_rules() -> list[Instrument]:
    """Extends Phase 0's single rule to the Phase 1 scenario scope
    (SCOUT.md #5): cancel_reservation + update_reservation_flights, both
    gated on a prior get_reservation_details lookup for the same
    reservation_id."""
    return [
        RequireLookupBeforeCancel(),
        RequireLookupBeforeCancel(
            gated_tool="update_reservation_flights",
            required_lookup_tool="get_reservation_details",
            key_arg="reservation_id",
        ),
    ]


def retail_fast_rules() -> list[Instrument]:
    """Retail analog of `default_fast_rules` (SCOUT.md Phase 2c): gate the
    destructive order-level write tools on a prior `get_order_details` lookup
    for the same `order_id`. Retail policy is prose-only, so the trace-lookup
    is the only structural signal available — the policy judge catches the
    rest (user-level PII / address writes are deliberately left to it)."""
    return [
        RequireLookupBeforeCancel(
            gated_tool="cancel_pending_order",
            required_lookup_tool="get_order_details",
            key_arg="order_id",
        ),
        RequireLookupBeforeCancel(
            gated_tool="return_delivered_order_items",
            required_lookup_tool="get_order_details",
            key_arg="order_id",
        ),
        RequireLookupBeforeCancel(
            gated_tool="modify_pending_order_payment",
            required_lookup_tool="get_order_details",
            key_arg="order_id",
        ),
    ]


@dataclass(frozen=True)
class ScoredStep:
    scenario_id: str
    step_index: int
    step: Step
    decision: Decision
    verdicts: list[InstrumentVerdict]


def _verdict_label(verdicts: list[InstrumentVerdict], instrument: str) -> str | None:
    return next((v.label for v in verdicts if v.instrument == instrument), None)


def run_scenario(
    scenario: Scenario,
    slow_instruments: list[SlowInstrument],
    fast_rules: list[Instrument] | None = None,
) -> tuple[Trace, list[ScoredStep]]:
    """Runs one scripted scenario through the two-speed GATE: fast rules
    decide allow/block synchronously per step; slow instruments (drift,
    policy) annotate concurrently. Per-step verdicts land in
    `Step.metadata[VERDICT_METADATA_KEY]` (SCOUT.md Phase 1 #4)."""
    gate = Gate(instruments=fast_rules if fast_rules is not None else default_fast_rules())
    two_speed = TwoSpeedGate(gate=gate, slow_instruments=slow_instruments)

    scored_steps: list[ScoredStep] = []
    steps: list[Step] = []
    try:
        for index, scenario_step in enumerate(scenario.steps):
            decision, verdicts_future = two_speed.process(scenario_step.proposed)
            verdicts = verdicts_future.result()

            step = make_step(
                trace_id=scenario.scenario_id,
                proposed=scenario_step.proposed,
                decision=decision,
                step_id=f"{scenario.scenario_id}-step-{index}",
            )
            step.metadata[VERDICT_METADATA_KEY] = {
                "drift": _verdict_label(verdicts, "drift"),
                "policy": _verdict_label(verdicts, "policy"),
            }
            steps.append(step)
            scored_steps.append(
                ScoredStep(
                    scenario_id=scenario.scenario_id,
                    step_index=index,
                    step=step,
                    decision=decision,
                    verdicts=verdicts,
                )
            )
    finally:
        two_speed.shutdown()

    trace = build_trace(
        trace_id=scenario.scenario_id, agent_config_ref="bossyk-sandbox-phase1@0.1", steps=steps
    )
    return trace, scored_steps


def to_membership(
    scored_steps: list[ScoredStep], outcome_lookup: OutcomeKeyLookup
) -> list[StepMembership]:
    """Joins scored steps against the A1 ground truth to build the 3-way
    orthogonality input. Steps with no A1 key are skipped."""
    memberships = []
    for scored in scored_steps:
        is_violation = outcome_lookup.is_violation(scored.scenario_id, scored.step_index)
        if is_violation is None:
            continue
        drift_label = _verdict_label(scored.verdicts, "drift")
        policy_label = _verdict_label(scored.verdicts, "policy")
        memberships.append(
            StepMembership(
                step_id=scored.step.step_id,
                drift_fires=drift_label in FIRING_LABELS,
                policy_fires=policy_label in FIRING_LABELS,
                outcome_violation=is_violation,
            )
        )
    return memberships


def to_gate_outcomes(
    scored_steps: list[ScoredStep], outcome_lookup: OutcomeKeyLookup
) -> list[GateOutcomeRecord]:
    """Joins scored steps against the A1 ground truth to build the
    gate-vs-ground-truth confusion input. Steps with no A1 key are skipped."""
    records = []
    for scored in scored_steps:
        is_violation = outcome_lookup.is_violation(scored.scenario_id, scored.step_index)
        if is_violation is None:
            continue
        records.append(
            GateOutcomeRecord(
                step_id=scored.step.step_id,
                gate_blocked=scored.decision.verdict is Verdict.BLOCK,
                ground_truth_violation=is_violation,
            )
        )
    return records
