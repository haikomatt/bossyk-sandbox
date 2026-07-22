from __future__ import annotations

from pathlib import Path

import bossyk_sandbox
from bossyk_sandbox.instruments.outcome_key import BoundaryLabel
from bossyk_sandbox.scenarios.loader import load_scenarios

RETAIL_SCENARIOS_PATH = (
    Path(bossyk_sandbox.__file__).parent / "scenarios" / "retail" / "scenarios.json"
)

# Real tau2 retail tool names (src/tau2/domains/retail/tools.py) — the
# retail scenario set must use these, not airline's cancel_reservation /
# update_reservation_flights / get_reservation_details.
KNOWN_RETAIL_TOOLS = {
    "calculate",
    "cancel_pending_order",
    "exchange_delivered_order_items",
    "find_user_id_by_name_zip",
    "find_user_id_by_email",
    "get_order_details",
    "get_product_details",
    "get_item_details",
    "get_user_details",
    "list_all_product_types",
    "modify_pending_order_address",
    "modify_pending_order_items",
    "modify_pending_order_payment",
    "modify_user_address",
    "return_delivered_order_items",
    "transfer_to_human_agents",
}

KNOWN_AIRLINE_TOOLS = {
    "cancel_reservation",
    "get_reservation_details",
    "update_reservation_flights",
}

EXPECTED_BOUNDARIES = {
    "cancel_without_auth",
    "refund_over_threshold",
    "pii_disclosure",
    "unauthorized_modification",
}


def _load() -> list:
    return load_scenarios(RETAIL_SCENARIOS_PATH)


def test_retail_scenario_set_has_at_least_eight_scenarios() -> None:
    scenarios = _load()
    assert len(scenarios) >= 8


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
    """Guards the byte-identical-intent bug (docs/drift-diagnostic-findings.md
    bug 2): each step's declared_intent must be its own, not a stale copy of
    an earlier step's — this is what let the two airline deviation scenarios
    slip past drift undetected."""
    scenarios = _load()
    for scenario in scenarios:
        intents = [step.proposed.declared_intent for step in scenario.steps]
        assert len(intents) == len(set(intents)), (
            f"{scenario.scenario_id} has duplicate declared_intent across steps: {intents}"
        )


def test_at_least_one_scenario_of_each_boundary_label_exists() -> None:
    scenarios = _load()
    labels = {step.boundary_label for scenario in scenarios for step in scenario.steps}
    assert labels == {
        BoundaryLabel.BENIGN,
        BoundaryLabel.POLICY_VIOLATION,
        BoundaryLabel.DEVIATION,
    }


def test_tool_names_are_retail_tools_not_airline_tools() -> None:
    scenarios = _load()
    tool_names = {step.proposed.tool_name for scenario in scenarios for step in scenario.steps}
    assert tool_names & KNOWN_AIRLINE_TOOLS == set()
    assert tool_names <= KNOWN_RETAIL_TOOLS, (
        f"Unknown retail tool(s): {tool_names - KNOWN_RETAIL_TOOLS}"
    )


def test_scenario_set_spans_all_four_retail_boundaries() -> None:
    """Each scenario_id encodes which of the four retail consequence
    boundaries (conditions/grid.py RETAIL_BOUNDARIES) it exercises via its
    slug; a loose substring check keeps this decoupled from exact wording."""
    scenarios = _load()
    slugs = " ".join(scenario.scenario_id for scenario in scenarios)
    assert "cancel" in slugs
    assert "refund" in slugs or "threshold" in slugs
    assert "pii" in slugs
    assert "modification" in slugs or "modif" in slugs


def test_at_least_two_deviation_scenarios_exist() -> None:
    """Mirrors airline's 004/008 scope-deviation pattern (declared_intent !=
    action) — need at least two to give the orthogonality table signal on
    the deviation axis."""
    scenarios = _load()
    deviation_scenarios = [
        scenario
        for scenario in scenarios
        if any(step.boundary_label == BoundaryLabel.DEVIATION for step in scenario.steps)
    ]
    assert len(deviation_scenarios) >= 2
