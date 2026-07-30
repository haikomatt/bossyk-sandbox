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
    gated_tool: str | None
    steps: list[ScenarioStep]
    # Slice 2, P6 (boundary 5, prohibited_financial_promotion): the spoken/
    # sent text an utterance-only scenario scores -- set only when there is
    # no `gated_tool` at all (Matt's chosen relaxation: relax the loader,
    # not a pseudo-step). `None` for every ordinary tool-call scenario.
    utterance: str | None = None


def load_scenarios(path: Path | str = SCENARIOS_PATH) -> list[Scenario]:
    """Loads the scripted airline scenario set + A1 boundary-outcome keys.
    Hand-authored (see SCOUT.md Phase 1 #5) — tau2's `evaluation_criteria`
    informed drafting but isn't parsed as a drop-in oracle.

    Slice 2, P6: a scenario may instead be utterance-only (boundary 5 has no
    tool call at all) -- `gated_tool` is then absent/`None` and `utterance`
    carries the text to score; `steps` may be empty in that case. An
    EXPLICIT empty-string `gated_tool` (`""`) is still an authoring error,
    distinct from an absent key: only a genuinely missing `gated_tool` means
    "utterance scenario". A scenario with neither is rejected outright."""
    data = json.loads(Path(path).read_text())
    scenarios = []
    seen_scenario_ids: set[str] = set()
    for entry in data["scenarios"]:
        scenario_id = entry["scenario_id"]
        if scenario_id in seen_scenario_ids:
            raise ValueError(f"duplicate scenario_id: {scenario_id!r}")
        seen_scenario_ids.add(scenario_id)

        gated_tool = entry.get("gated_tool")
        utterance = entry.get("utterance")

        if gated_tool is not None and not gated_tool:
            raise ValueError(f"{scenario_id!r} has an empty gated_tool")

        if gated_tool is None and not utterance:
            raise ValueError(
                f"{scenario_id!r} has neither a gated_tool nor an utterance -- nothing to score"
            )

        raw_steps = entry.get("steps") or []
        if gated_tool is not None and not raw_steps:
            raise ValueError(f"{scenario_id!r} has no steps")

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

        if gated_tool is not None and gated_tool not in {step.proposed.tool_name for step in steps}:
            raise ValueError(
                f"{scenario_id!r} has gated_tool {gated_tool!r} that is not among its "
                "steps' tool_names"
            )

        scenarios.append(
            Scenario(
                scenario_id=scenario_id, gated_tool=gated_tool, steps=steps, utterance=utterance
            )
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
