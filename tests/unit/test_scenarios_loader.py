from __future__ import annotations

from bossyk_sandbox.instruments.outcome_key import BoundaryLabel
from bossyk_sandbox.scenarios.loader import load_scenarios, outcome_keys


def test_scenario_set_has_eight_to_twelve_scenarios() -> None:
    scenarios = load_scenarios()
    assert 8 <= len(scenarios) <= 12


def test_scenario_set_covers_both_gated_tools() -> None:
    scenarios = load_scenarios()
    gated_tools = {scenario.gated_tool for scenario in scenarios}
    assert gated_tools == {"cancel_reservation", "update_reservation_flights"}


def test_scenario_set_spans_all_three_boundary_labels() -> None:
    scenarios = load_scenarios()
    labels = {step.boundary_label for scenario in scenarios for step in scenario.steps}
    assert labels == {BoundaryLabel.BENIGN, BoundaryLabel.POLICY_VIOLATION, BoundaryLabel.DEVIATION}


def test_outcome_keys_cover_every_step() -> None:
    scenarios = load_scenarios()
    keys = outcome_keys(scenarios)
    total_steps = sum(len(scenario.steps) for scenario in scenarios)
    assert len(keys) == total_steps


def test_every_step_has_a_declared_intent() -> None:
    scenarios = load_scenarios()
    for scenario in scenarios:
        for step in scenario.steps:
            assert step.proposed.declared_intent, (
                f"{scenario.scenario_id} step missing declared_intent"
            )


def test_declared_intent_is_unique_per_step_within_a_scenario() -> None:
    """Each step's declared_intent must be its own, not a stale copy of an
    earlier step's (see docs/drift-diagnostic-findings.md bug 2: the two
    `deviation` scenarios previously duplicated step 0's declared_intent
    verbatim on step 1)."""
    scenarios = load_scenarios()
    for scenario in scenarios:
        intents = [step.proposed.declared_intent for step in scenario.steps]
        assert len(intents) == len(set(intents)), (
            f"{scenario.scenario_id} has duplicate declared_intent across steps: {intents}"
        )
