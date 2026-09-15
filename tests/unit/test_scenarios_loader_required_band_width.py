"""RED-phase tests for the scenario loader's `required_band_width` field
(bossyk-sandbox privacy/minimisation demonstrator, Phase 2).

Convention (documented on `Scenario.required_band_width`): the width, in
GBP, of the tax band the scenario's task genuinely requires -- i.e. the band
containing the persona's true income, per `bossyk_sandbox.advice.tax`. A
scenario that never exercises the free-threshold predicate has no opinion on
this and the field is simply absent/`None`; existing domains (airline,
retail, outreach) are unaffected -- the field is optional and defaults to
`None` so their scenario JSON needs no change.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from bossyk_sandbox.scenarios.loader import load_scenarios


def _write_fixture(tmp_path: Path, scenarios: list[dict[str, Any]]) -> Path:
    path = tmp_path / "scenarios.json"
    path.write_text(json.dumps({"scenarios": scenarios}))
    return path


def _minimal_entry(scenario_id: str, **extra: Any) -> dict[str, Any]:
    return {
        "scenario_id": scenario_id,
        "gated_tool": "get_tax_position",
        "steps": [
            {
                "tool_name": "get_tax_position",
                "arguments": {"ref": "ADV-0001"},
                "declared_intent": "check band",
                "boundary_label": "benign",
            }
        ],
        **extra,
    }


def test_required_band_width_defaults_to_none_when_absent(tmp_path: Path) -> None:
    path = _write_fixture(tmp_path, [_minimal_entry("s1")])

    scenarios = load_scenarios(path)

    assert scenarios[0].required_band_width is None


def test_required_band_width_is_parsed_when_present(tmp_path: Path) -> None:
    path = _write_fixture(tmp_path, [_minimal_entry("s1", required_band_width=50270)])

    scenarios = load_scenarios(path)

    assert scenarios[0].required_band_width == 50270


def test_required_band_width_is_parsed_as_a_float(tmp_path: Path) -> None:
    path = _write_fixture(tmp_path, [_minimal_entry("s1", required_band_width=74870)])

    scenarios = load_scenarios(path)

    assert isinstance(scenarios[0].required_band_width, float)
    assert scenarios[0].required_band_width == 74870.0
