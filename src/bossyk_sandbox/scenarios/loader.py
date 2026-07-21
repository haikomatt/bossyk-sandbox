from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from bossyk_sandbox.instruments.base import ProposedAction
from bossyk_sandbox.instruments.outcome_key import BoundaryLabel, OutcomeKey

SCENARIOS_PATH = Path(__file__).parent / "airline" / "scenarios.json"


@dataclass(frozen=True)
class ScenarioStep:
    proposed: ProposedAction
    boundary_label: BoundaryLabel


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    gated_tool: str
    steps: list[ScenarioStep]


def load_scenarios(path: Path | str = SCENARIOS_PATH) -> list[Scenario]:
    """Loads the scripted airline scenario set + A1 boundary-outcome keys.
    Hand-authored (see SCOUT.md Phase 1 #5) — tau2's `evaluation_criteria`
    informed drafting but isn't parsed as a drop-in oracle."""
    data = json.loads(Path(path).read_text())
    scenarios = []
    for entry in data["scenarios"]:
        steps = [
            ScenarioStep(
                proposed=ProposedAction(
                    tool_name=step["tool_name"],
                    arguments=step["arguments"],
                    declared_intent=step.get("declared_intent"),
                ),
                boundary_label=BoundaryLabel(step["boundary_label"]),
            )
            for step in entry["steps"]
        ]
        scenarios.append(
            Scenario(scenario_id=entry["scenario_id"], gated_tool=entry["gated_tool"], steps=steps)
        )
    return scenarios


def outcome_keys(scenarios: list[Scenario]) -> list[OutcomeKey]:
    return [
        OutcomeKey(
            scenario_id=scenario.scenario_id, step_index=i, boundary_label=step.boundary_label
        )
        for scenario in scenarios
        for i, step in enumerate(scenario.steps)
    ]
