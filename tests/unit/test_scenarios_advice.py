from __future__ import annotations

from pathlib import Path

import bossyk_sandbox
from bossyk_sandbox.instruments.outcome_key import BoundaryLabel
from bossyk_sandbox.scenarios.loader import Scenario, load_scenarios

ADVICE_SCENARIOS_PATH = (
    Path(bossyk_sandbox.__file__).parent / "scenarios" / "advice" / "scenarios.json"
)

KNOWN_ADVICE_TOOLS = {
    "get_customer_profile",
    "get_tax_position",
    "get_contribution_headroom",
    "is_income_above",
}


def _load() -> list[Scenario]:
    return load_scenarios(ADVICE_SCENARIOS_PATH)


def test_advice_scenario_set_has_exactly_two_scenarios() -> None:
    # Phase 1 scope: exactly one benign band-only task and one
    # precision-rewarding task that tempts raw/predicate use -- no more.
    scenarios = _load()
    assert len(scenarios) == 2


def test_every_step_has_a_non_empty_declared_intent() -> None:
    scenarios = _load()
    for scenario in scenarios:
        for step in scenario.steps:
            assert step.proposed.declared_intent, (
                f"{scenario.scenario_id} step missing declared_intent"
            )


def test_every_boundary_label_is_a_valid_boundary_label() -> None:
    scenarios = _load()
    valid_labels = {label.value for label in BoundaryLabel}
    for scenario in scenarios:
        for step in scenario.steps:
            assert step.boundary_label.value in valid_labels


def test_declared_intent_is_unique_per_step_within_a_scenario() -> None:
    scenarios = _load()
    for scenario in scenarios:
        intents = [step.proposed.declared_intent for step in scenario.steps]
        assert len(intents) == len(set(intents)), (
            f"{scenario.scenario_id} has duplicate declared_intent across steps: {intents}"
        )


def test_tool_names_are_known_advice_tools() -> None:
    scenarios = _load()
    tool_names = {step.proposed.tool_name for scenario in scenarios for step in scenario.steps}
    assert tool_names <= KNOWN_ADVICE_TOOLS, (
        f"Unknown advice tool(s): {tool_names - KNOWN_ADVICE_TOOLS}"
    )


def test_benign_scenario_is_answerable_without_the_raw_profile_tool() -> None:
    scenarios = _load()
    benign = next(s for s in scenarios if s.scenario_id.startswith("advice-001"))
    for step in benign.steps:
        assert step.proposed.tool_name != "get_customer_profile"


def test_precision_scenario_includes_a_raw_profile_call() -> None:
    scenarios = _load()
    precision = next(s for s in scenarios if s.scenario_id.startswith("advice-002"))
    tool_names = {step.proposed.tool_name for step in precision.steps}
    assert "get_customer_profile" in tool_names
