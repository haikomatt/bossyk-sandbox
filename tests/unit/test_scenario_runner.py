from __future__ import annotations

from bossyk_sandbox.evidence.trace import make_step
from bossyk_sandbox.instruments.base import Decision, InstrumentVerdict, ProposedAction, Verdict
from bossyk_sandbox.instruments.outcome_key import BoundaryLabel, OutcomeKey, OutcomeKeyLookup
from bossyk_sandbox.scenarios.loader import Scenario, ScenarioStep
from bossyk_sandbox.scenarios.runner import (
    VERDICT_METADATA_KEY,
    InstrumentAvailability,
    ScoredStep,
    instrument_availability,
    run_scenario,
    to_gate_outcomes,
    to_membership,
    verdict_state,
)


class _FixedInstrument:
    def __init__(self, name: str, label: str, detail: str = "") -> None:
        self.name = name
        self._label = label
        self._detail = detail

    def annotate(
        self, proposed: ProposedAction, history: list[ProposedAction]
    ) -> InstrumentVerdict:
        return InstrumentVerdict(instrument=self.name, label=self._label, detail=self._detail)


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
        assert verdict_metadata == {
            "drift": {"label": "faithful", "detail": ""},
            "policy": {"label": "instruction_noncompliance", "detail": ""},
        }


# --- Finding 12: evidence metadata must reflect whatever slow instruments
# are actually configured, not a hardcoded drift/policy shape -------------


def test_run_scenario_records_every_configured_instrument_with_label_and_detail() -> None:
    """`run_scenario` currently hardcodes
    `{"drift": ..., "policy": ...}` in `Step.metadata[VERDICT_METADATA_KEY]`,
    so a third configured instrument ("custom") scores but never lands in
    evidence, and only the label (not the detail) survives. The metadata
    must instead be keyed by instrument name, one entry per *configured*
    slow instrument, each carrying both label and detail."""
    scenario = _two_step_scenario()
    drift = _FixedInstrument("drift", "faithful", detail="drift detail")
    policy = _FixedInstrument("policy", "instruction_noncompliance", detail="policy detail")
    custom = _FixedInstrument("custom", "goal_deviation", detail="custom detail")

    _trace, scored_steps = run_scenario(scenario, slow_instruments=[drift, policy, custom])

    assert len(scored_steps) == 2
    for scored in scored_steps:
        verdict_metadata = scored.step.metadata[VERDICT_METADATA_KEY]
        assert verdict_metadata == {
            "drift": {"label": "faithful", "detail": "drift detail"},
            "policy": {"label": "instruction_noncompliance", "detail": "policy detail"},
            "custom": {"label": "goal_deviation", "detail": "custom detail"},
        }


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


# --- verdict_state / tri-state membership (Finding 4) -----------------------


def test_verdict_state_maps_error_label_to_none() -> None:
    assert verdict_state("error") is None


def test_verdict_state_maps_unscored_label_to_none() -> None:
    assert verdict_state("unscored") is None


def test_verdict_state_maps_missing_label_to_none() -> None:
    assert verdict_state(None) is None


def test_verdict_state_maps_firing_label_to_true() -> None:
    assert verdict_state("goal_deviation") is True


def test_verdict_state_maps_non_firing_label_to_false() -> None:
    assert verdict_state("faithful") is False


def test_to_membership_error_drift_verdict_yields_none_not_false() -> None:
    # Finding 4: a judge that errored on this step must not be reported as
    # "did not fire" -- that silently deflates the drift detection rate.
    scenario = _two_step_scenario()
    drift = _FixedInstrument("drift", "error")
    policy = _FixedInstrument("policy", "goal_deviation")
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

    assert memberships[0].drift_fires is None
    assert memberships[0].policy_fires is True


# --- instrument_availability (Finding 4) -------------------------------------


def _scored_step(step_index: int, verdicts: list[InstrumentVerdict]) -> ScoredStep:
    proposed = ProposedAction("get_reservation_details", {"reservation_id": "R1"})
    decision = Decision(Verdict.ALLOW, "ok")
    step = make_step(
        trace_id="availability-scenario",
        proposed=proposed,
        decision=decision,
        step_id=f"availability-scenario-step-{step_index}",
    )
    return ScoredStep(
        scenario_id="availability-scenario",
        step_index=step_index,
        step=step,
        decision=decision,
        verdicts=verdicts,
    )


def test_instrument_availability_counts_scored_error_unscored_and_missing_steps() -> None:
    scored_steps = [
        _scored_step(0, [InstrumentVerdict(instrument="drift", label="goal_deviation")]),
        _scored_step(1, [InstrumentVerdict(instrument="drift", label="error")]),
        _scored_step(2, [InstrumentVerdict(instrument="drift", label="unscored")]),
        _scored_step(3, [InstrumentVerdict(instrument="policy", label="goal_deviation")]),
    ]

    availability = instrument_availability(scored_steps, "drift")

    assert availability == InstrumentAvailability(
        instrument="drift", n_steps=4, n_scored=1, n_error=1, n_unscored=1, n_missing=1
    )
