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
    seen_scenario_ids: set[str] = set()
    for entry in data["scenarios"]:
        scenario_id = entry["scenario_id"]
        if scenario_id in seen_scenario_ids:
            raise ValueError(f"duplicate scenario_id: {scenario_id!r}")
        seen_scenario_ids.add(scenario_id)

        raw_steps = entry["steps"]
        if not raw_steps:
            raise ValueError(f"{scenario_id!r} has no steps")

        gated_tool = entry["gated_tool"]
        if not gated_tool:
            raise ValueError(f"{scenario_id!r} has an empty gated_tool")

        steps = [
            ScenarioStep(
                proposed=ProposedAction(
                    tool_name=step["tool_name"],
                    arguments=step["arguments"],
                    declared_intent=step.get("declared_intent"),
                ),
                boundary_label=BoundaryLabel(step["boundary_label"]),
            )
            for step in raw_steps
        ]

        if gated_tool not in {step.proposed.tool_name for step in steps}:
            raise ValueError(
                f"{scenario_id!r} has gated_tool {gated_tool!r} that is not among its "
                "steps' tool_names"
            )

        scenarios.append(Scenario(scenario_id=scenario_id, gated_tool=gated_tool, steps=steps))
    return scenarios


def outcome_keys(scenarios: list[Scenario]) -> list[OutcomeKey]:
    return [
        OutcomeKey(
            scenario_id=scenario.scenario_id, step_index=i, boundary_label=step.boundary_label
        )
        for scenario in scenarios
        for i, step in enumerate(scenario.steps)
    ]
