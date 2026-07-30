"""RED-phase tests for `scenarios/outreach/scenarios.json` (bossyk-sandbox
slice 1), loaded through the existing (unmodified) scenario loader.

Slice 1 scope: only the two PRECEDENCE-shaped boundaries are functional this
phase -- `booking_without_eligibility` (book_survey gated on a prior
check_eligibility) and `unauthorised_incentive` (apply_discount gated on
standing max_amount). contact_without_consent, out_of_hours_contact,
prohibited_financial_promotion and pii_cross_contact are slice 2/3 work; this
phase deliberately does NOT author scenario data for them, so every scenario
has a real gated tool call and the loader's existing
gated_tool-must-match-a-step invariant (loader.py:59-63) is never hit this
phase -- see outreach-domain-cleanroom-spec.md "Open item for the phase gate"
for the utterance-only-scenario question that is intentionally deferred.

This file collects fine today (imports nothing new); it fails at test run
time with FileNotFoundError because scenarios/outreach/scenarios.json does
not exist yet.
"""

from __future__ import annotations

from pathlib import Path

import bossyk_sandbox
from bossyk_sandbox.instruments.outcome_key import BoundaryLabel
from bossyk_sandbox.scenarios.loader import Scenario, load_scenarios

OUTREACH_SCENARIOS_PATH = (
    Path(bossyk_sandbox.__file__).parent / "scenarios" / "outreach" / "scenarios.json"
)

KNOWN_OUTREACH_TOOLS = {
    "lookup_prospect",
    "check_suppression",
    "check_eligibility",
    "get_quote",
    "place_call",
    "send_sms",
    "send_email",
    "book_survey",
    "apply_discount",
    "record_consent",
}


def _load() -> list[Scenario]:
    return load_scenarios(OUTREACH_SCENARIOS_PATH)


def test_outreach_scenario_file_loads_at_least_four_scenarios() -> None:
    scenarios = _load()
    assert len(scenarios) >= 4


def test_outreach_scenario_set_covers_both_slice_1_gated_tools() -> None:
    scenarios = _load()
    gated_tools = {scenario.gated_tool for scenario in scenarios}
    assert gated_tools == {"book_survey", "apply_discount"}


def test_every_step_has_a_non_empty_declared_intent() -> None:
    scenarios = _load()
    for scenario in scenarios:
        for step in scenario.steps:
            assert step.proposed.declared_intent, (
                f"{scenario.scenario_id} step missing declared_intent"
            )


def test_declared_intent_is_unique_per_step_within_a_scenario() -> None:
    scenarios = _load()
    for scenario in scenarios:
        intents = [step.proposed.declared_intent for step in scenario.steps]
        assert len(intents) == len(set(intents)), (
            f"{scenario.scenario_id} has duplicate declared_intent across steps: {intents}"
        )


def test_tool_names_are_known_outreach_tools() -> None:
    scenarios = _load()
    tool_names = {step.proposed.tool_name for scenario in scenarios for step in scenario.steps}
    assert tool_names <= KNOWN_OUTREACH_TOOLS


def test_scenario_set_has_a_benign_and_a_policy_violation_booking_scenario() -> None:
    scenarios = _load()
    booking_scenarios = [s for s in scenarios if s.gated_tool == "book_survey"]
    labels = {step.boundary_label for scenario in booking_scenarios for step in scenario.steps}
    assert BoundaryLabel.BENIGN in labels
    assert BoundaryLabel.POLICY_VIOLATION in labels


def test_scenario_set_has_a_benign_and_a_policy_violation_discount_scenario() -> None:
    scenarios = _load()
    discount_scenarios = [s for s in scenarios if s.gated_tool == "apply_discount"]
    labels = {step.boundary_label for scenario in discount_scenarios for step in scenario.steps}
    assert BoundaryLabel.BENIGN in labels
    assert BoundaryLabel.POLICY_VIOLATION in labels


def test_booking_violation_scenario_has_no_prior_eligibility_check() -> None:
    scenarios = _load()
    violation = next(
        s
        for s in scenarios
        if s.gated_tool == "book_survey"
        and any(step.boundary_label == BoundaryLabel.POLICY_VIOLATION for step in s.steps)
    )
    tool_names = [step.proposed.tool_name for step in violation.steps]
    assert "check_eligibility" not in tool_names
    assert "book_survey" in tool_names
