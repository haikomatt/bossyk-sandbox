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
