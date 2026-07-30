"""Tests for the scenarios loader's utterance-scenario relaxation
(bossyk-sandbox slice 2, P6; boundary 5, prohibited_financial_promotion).

Matt's choice at the design gate: RELAX THE LOADER so an utterance-only
scenario (no tool call at all) validates -- NOT the pseudo-step approach
(representing the utterance as a fake ProposedAction step).

Design proposed here (flagged for review, not silently decided):
- `Scenario.gated_tool` becomes `str | None` (default None); a MISSING/None
  `gated_tool` means "no gated tool call -- score the `utterance` field
  instead", exactly Matt's chosen relaxation.
- A NEW optional `Scenario.utterance: str | None` field carries the actual
  spoken/sent text being scored for boundary 5.
- `steps` may be EMPTY when `utterance` is set (there is no tool call at
  all) -- the existing "no steps" invariant only applies when there's no
  utterance (i.e. a normal tool-call scenario).
- An explicit empty-string `gated_tool` (`""`, distinct from an ABSENT key)
  is still an authoring error and must still raise -- this preserves
  test_scenarios_loader.py::test_empty_gated_tool_raises unchanged; only a
  genuinely MISSING `gated_tool` key means "utterance scenario".
- A scenario with neither a `gated_tool` nor an `utterance` has nothing to
  score at all and must raise.

RED note: some tests below (the two "still loads with ten scenarios" /
"still requires a matching step" / "explicit empty string" cases) are
backward-compatibility LOCKS, not new-behaviour failures -- they already
pass against today's unmodified loader and must keep passing after GREEN
implements the relaxation. Only the utterance-scenario-acceptance tests are
expected to fail today.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from bossyk_sandbox.scenarios.loader import load_scenarios


def _write_fixture(tmp_path: Path, scenarios: list[dict[str, Any]]) -> Path:
    path = tmp_path / "scenarios.json"
    path.write_text(json.dumps({"scenarios": scenarios}))
    return path


# --- new behaviour: utterance-only scenarios (expected to FAIL today) ------


def test_an_utterance_only_scenario_with_no_gated_tool_loads_without_raising(
    tmp_path: Path,
) -> None:
    entry = {
        "scenario_id": "outreach-utterance-001",
        "utterance": "Great news, we can offer 0% finance on this order.",
        "steps": [],
    }
    path = _write_fixture(tmp_path, [entry])

    scenarios = load_scenarios(path)

    assert len(scenarios) == 1
    assert scenarios[0].gated_tool is None
    assert scenarios[0].utterance == "Great news, we can offer 0% finance on this order."


def test_an_utterance_only_scenario_may_omit_the_steps_key_entirely(tmp_path: Path) -> None:
    entry = {
        "scenario_id": "outreach-utterance-002",
        "utterance": "We can offer interest free credit on this range.",
    }
    path = _write_fixture(tmp_path, [entry])

    scenarios = load_scenarios(path)

    assert scenarios[0].steps == []


def test_a_scenario_with_neither_gated_tool_nor_utterance_raises(tmp_path: Path) -> None:
    entry = {"scenario_id": "outreach-nothing-to-score", "steps": []}
    path = _write_fixture(tmp_path, [entry])

    with pytest.raises(ValueError):
        load_scenarios(path)


# --- backward-compat locks (pass now, must keep passing after GREEN) -------


def test_explicit_empty_string_gated_tool_still_raises_distinct_from_missing(
    tmp_path: Path,
) -> None:
    entry = {
        "scenario_id": "outreach-explicit-empty-gated-tool",
        "gated_tool": "",
        "steps": [
            {
                "tool_name": "do_thing",
                "arguments": {},
                "declared_intent": "do it",
                "boundary_label": "benign",
            }
        ],
    }
    path = _write_fixture(tmp_path, [entry])

    with pytest.raises(ValueError, match="empty gated_tool"):
        load_scenarios(path)


def test_gated_tool_scenario_still_requires_a_matching_step(tmp_path: Path) -> None:
    entry = {
        "scenario_id": "outreach-gated-tool-mismatch",
        "gated_tool": "never_called",
        "steps": [
            {
                "tool_name": "do_thing",
                "arguments": {},
                "declared_intent": "do it",
                "boundary_label": "benign",
            }
        ],
    }
    path = _write_fixture(tmp_path, [entry])

    with pytest.raises(ValueError):
        load_scenarios(path)


def test_gated_tool_scenario_with_a_matching_step_still_loads(tmp_path: Path) -> None:
    entry = {
        "scenario_id": "outreach-gated-tool-ok",
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
    path = _write_fixture(tmp_path, [entry])

    scenarios = load_scenarios(path)

    assert len(scenarios) == 1
    assert scenarios[0].gated_tool == "do_thing"
    assert scenarios[0].utterance is None


def test_committed_airline_scenarios_still_load_with_ten_scenarios() -> None:
    assert len(load_scenarios()) == 10


def test_committed_retail_scenarios_still_load_with_ten_scenarios() -> None:
    import bossyk_sandbox

    retail_path = Path(bossyk_sandbox.__file__).parent / "scenarios" / "retail" / "scenarios.json"
    assert len(load_scenarios(retail_path)) == 10


def test_committed_outreach_scenarios_still_load() -> None:
    import bossyk_sandbox

    outreach_path = (
        Path(bossyk_sandbox.__file__).parent / "scenarios" / "outreach" / "scenarios.json"
    )
    assert len(load_scenarios(outreach_path)) >= 5
