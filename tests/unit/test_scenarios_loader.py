from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import bossyk_sandbox
from bossyk_sandbox.instruments.outcome_key import BoundaryLabel
from bossyk_sandbox.scenarios.loader import load_scenarios, outcome_keys

RETAIL_SCENARIOS_PATH = (
    Path(bossyk_sandbox.__file__).parent / "scenarios" / "retail" / "scenarios.json"
)


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


# --- ground-truth validation (Finding 11) ------------------------------------
#
# `load_scenarios` today parses the JSON structurally but validates nothing
# about it: a duplicate scenario_id, a scenario with no steps, or a
# gated_tool that doesn't name any of the scenario's own tool calls all load
# silently. An unknown boundary_label already raises today via the
# `BoundaryLabel` enum -- that behavior is locked in below, not changed.


def _valid_entry(**overrides: Any) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "scenario_id": "s1",
        "gated_tool": "do_thing",
        "steps": [
            {
                "tool_name": "do_thing",
                "arguments": {},
                "declared_intent": "do it",
                "boundary_label": "benign",
            }
        ],
    }
    entry.update(overrides)
    return entry


def _write_fixture(tmp_path: Path, scenarios: list[dict[str, Any]]) -> Path:
    path = tmp_path / "scenarios.json"
    path.write_text(json.dumps({"scenarios": scenarios}))
    return path


def test_duplicate_scenario_id_raises(tmp_path: Path) -> None:
    path = _write_fixture(tmp_path, [_valid_entry(), _valid_entry()])

    with pytest.raises(ValueError):
        load_scenarios(path)


def test_scenario_with_empty_steps_raises(tmp_path: Path) -> None:
    path = _write_fixture(tmp_path, [_valid_entry(steps=[])])

    with pytest.raises(ValueError):
        load_scenarios(path)


def test_empty_gated_tool_raises(tmp_path: Path) -> None:
    path = _write_fixture(tmp_path, [_valid_entry(gated_tool="")])

    with pytest.raises(ValueError):
        load_scenarios(path)


def test_gated_tool_not_among_the_scenarios_step_tool_names_raises(tmp_path: Path) -> None:
    path = _write_fixture(tmp_path, [_valid_entry(gated_tool="never_called")])

    with pytest.raises(ValueError):
        load_scenarios(path)


def test_unknown_boundary_label_raises_value_error(tmp_path: Path) -> None:
    # Already true today via the BoundaryLabel enum -- locking it in.
    entry = _valid_entry()
    entry["steps"][0]["boundary_label"] = "not_a_real_boundary_label"
    path = _write_fixture(tmp_path, [entry])

    with pytest.raises(ValueError):
        load_scenarios(path)


def test_committed_airline_scenarios_still_load_with_ten_scenarios() -> None:
    assert len(load_scenarios()) == 10


def test_committed_retail_scenarios_still_load_with_ten_scenarios() -> None:
    assert len(load_scenarios(RETAIL_SCENARIOS_PATH)) == 10
