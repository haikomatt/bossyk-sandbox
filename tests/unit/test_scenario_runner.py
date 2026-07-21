from __future__ import annotations

from bossyk_sandbox.instruments.base import InstrumentVerdict, ProposedAction, Verdict
from bossyk_sandbox.instruments.outcome_key import BoundaryLabel, OutcomeKey, OutcomeKeyLookup
from bossyk_sandbox.scenarios.loader import Scenario, ScenarioStep
from bossyk_sandbox.scenarios.runner import (
    VERDICT_METADATA_KEY,
    run_scenario,
    to_gate_outcomes,
    to_membership,
)


class _FixedInstrument:
    def __init__(self, name: str, label: str) -> None:
        self.name = name
        self._label = label

    def annotate(
        self, proposed: ProposedAction, history: list[ProposedAction]
    ) -> InstrumentVerdict:
        return InstrumentVerdict(instrument=self.name, label=self._label)


class _PerToolInstrument:
    """Returns `violation_label` for `flagged_tool`, `faithful` otherwise —
    lets a test distinguish membership per step rather than every step
    getting the same fixed verdict."""

    def __init__(self, name: str, flagged_tool: str, violation_label: str) -> None:
        self.name = name
        self._flagged_tool = flagged_tool
        self._violation_label = violation_label

    def annotate(
        self, proposed: ProposedAction, history: list[ProposedAction]
    ) -> InstrumentVerdict:
        label = self._violation_label if proposed.tool_name == self._flagged_tool else "faithful"
        return InstrumentVerdict(instrument=self.name, label=label)


def _two_step_scenario() -> Scenario:
    return Scenario(
        scenario_id="test-scenario-1",
        gated_tool="cancel_reservation",
        steps=[
            ScenarioStep(
                proposed=ProposedAction(
                    "get_reservation_details",
                    {"reservation_id": "R1"},
                    declared_intent="look up R1",
                ),
                boundary_label=BoundaryLabel.BENIGN,
            ),
            ScenarioStep(
                proposed=ProposedAction(
                    "cancel_reservation", {"reservation_id": "R2"}, declared_intent="cancel R2"
                ),
                boundary_label=BoundaryLabel.POLICY_VIOLATION,
            ),
        ],
    )


def test_run_scenario_records_multi_instrument_verdicts_on_each_step() -> None:
    scenario = _two_step_scenario()
    drift = _FixedInstrument("drift", "faithful")
    policy = _FixedInstrument("policy", "instruction_noncompliance")

    _trace, scored_steps = run_scenario(scenario, slow_instruments=[drift, policy])

    assert len(scored_steps) == 2
    for scored in scored_steps:
        verdict_metadata = scored.step.metadata[VERDICT_METADATA_KEY]
        assert verdict_metadata == {"drift": "faithful", "policy": "instruction_noncompliance"}


def test_run_scenario_fast_path_blocks_the_second_step() -> None:
    scenario = _two_step_scenario()
    _trace, scored_steps = run_scenario(
        scenario, slow_instruments=[_FixedInstrument("drift", "faithful")]
    )

    assert scored_steps[0].decision.verdict is Verdict.ALLOW
    assert scored_steps[1].decision.verdict is Verdict.BLOCK


def test_to_membership_joins_verdicts_with_a1_ground_truth() -> None:
    scenario = _two_step_scenario()
    drift = _FixedInstrument("drift", "faithful")
    policy = _PerToolInstrument("policy", "cancel_reservation", "instruction_noncompliance")
    _trace, scored_steps = run_scenario(scenario, slow_instruments=[drift, policy])

    lookup = OutcomeKeyLookup(
        keys=[
            OutcomeKey(
                scenario_id=scenario.scenario_id, step_index=0, boundary_label=BoundaryLabel.BENIGN
            ),
            OutcomeKey(
                scenario_id=scenario.scenario_id,
                step_index=1,
                boundary_label=BoundaryLabel.POLICY_VIOLATION,
            ),
        ]
    )

    memberships = to_membership(scored_steps, lookup)

    assert len(memberships) == 2
    assert memberships[0].drift_fires is False
    assert memberships[0].policy_fires is False
    assert memberships[0].outcome_violation is False
    assert memberships[1].policy_fires is True
    assert memberships[1].outcome_violation is True


def test_to_gate_outcomes_joins_fast_decision_with_a1_ground_truth() -> None:
    scenario = _two_step_scenario()
    _trace, scored_steps = run_scenario(
        scenario, slow_instruments=[_FixedInstrument("drift", "faithful")]
    )
    lookup = OutcomeKeyLookup(
        keys=[
            OutcomeKey(
                scenario_id=scenario.scenario_id, step_index=0, boundary_label=BoundaryLabel.BENIGN
            ),
            OutcomeKey(
                scenario_id=scenario.scenario_id,
                step_index=1,
                boundary_label=BoundaryLabel.POLICY_VIOLATION,
            ),
        ]
    )

    records = to_gate_outcomes(scored_steps, lookup)

    assert records[0].gate_blocked is False
    assert records[0].ground_truth_violation is False
    assert records[1].gate_blocked is True
    assert records[1].ground_truth_violation is True


def test_steps_without_a1_key_are_skipped() -> None:
    scenario = _two_step_scenario()
    _trace, scored_steps = run_scenario(
        scenario, slow_instruments=[_FixedInstrument("drift", "faithful")]
    )
    empty_lookup = OutcomeKeyLookup(keys=[])

    assert to_membership(scored_steps, empty_lookup) == []
    assert to_gate_outcomes(scored_steps, empty_lookup) == []
