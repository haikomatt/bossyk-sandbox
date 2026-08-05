from __future__ import annotations

from pathlib import Path

import bossyk_sandbox
from bossyk_sandbox.instruments.outcome_key import BoundaryLabel
from bossyk_sandbox.scenarios.loader import Scenario, load_scenarios
from bossyk_sandbox.scenarios.runner import advice_eligibility_fast_rules, run_scenario

ELIGIBILITY_SCENARIOS_PATH = (
    Path(bossyk_sandbox.__file__).parent / "scenarios" / "advice" / "eligibility-scenarios.json"
)

# The eligibility surface (advice/toolkit.py) plus the existing advice read
# tools, kept available as compliant filler (spec: "keep the existing read
# tools as compliant filler").
KNOWN_ADVICE_ELIGIBILITY_TOOLS = {
    "verify_eligibility",
    "submit_eligibility_decision",
    "get_customer_profile",
    "get_tax_position",
    "get_contribution_headroom",
    "is_income_above",
}

# Lexically distant from retail (orders/refunds/returns) and airline
# (flights/bookings/rebooking) -- the distance lever the spec's vocabulary
# section calls for.
RETAIL_AIRLINE_VOCAB = {
    "order",
    "refund",
    "return",
    "flight",
    "booking",
    "reservation",
    "rebooking",
}


def _load() -> list[Scenario]:
    return load_scenarios(ELIGIBILITY_SCENARIOS_PATH)


def test_eligibility_scenario_set_has_at_least_four_scenarios() -> None:
    scenarios = _load()
    assert len(scenarios) >= 4


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


def test_tool_names_are_known_advice_eligibility_tools() -> None:
    scenarios = _load()
    tool_names = {step.proposed.tool_name for scenario in scenarios for step in scenario.steps}
    assert tool_names <= KNOWN_ADVICE_ELIGIBILITY_TOOLS, (
        f"Unknown advice-eligibility tool(s): {tool_names - KNOWN_ADVICE_ELIGIBILITY_TOOLS}"
    )


def test_both_benign_and_policy_violation_scenarios_exist() -> None:
    scenarios = _load()
    labels = {step.boundary_label for scenario in scenarios for step in scenario.steps}
    assert BoundaryLabel.BENIGN in labels
    assert BoundaryLabel.POLICY_VIOLATION in labels


def test_vocabulary_is_lexically_distant_from_retail_and_airline() -> None:
    """Guards the spec's distance lever at authoring time: no scenario text
    (declared_intent) should lean on retail/airline's core nouns. The full
    quantitative check is the pre-registered lexical-overlap gate
    (scripts/lexical_overlap_audit.py); this is a cheap textual tripwire."""
    scenarios = _load()
    text = " ".join(
        step.proposed.declared_intent or "" for scenario in scenarios for step in scenario.steps
    ).lower()
    for word in RETAIL_AIRLINE_VOCAB:
        assert word not in text, f"{word!r} found in advice-eligibility scenario text"


def test_gate_blocks_submit_first_and_allows_verify_then_submit() -> None:
    """The crossing this whole domain exists to produce, mirroring the
    retail rule tests (test_hardcoded_rule.py) but exercised end-to-end
    through `run_scenario` against the real authored scenario set: a
    `submit_eligibility_decision` with no prior `verify_eligibility` for the
    same `ref` is BLOCKed (policy_violation); verify-then-submit is ALLOWed
    (benign)."""
    scenarios = _load()
    fast_rules = advice_eligibility_fast_rules()

    for scenario in scenarios:
        _trace, scored_steps = run_scenario(scenario, slow_instruments=[], fast_rules=fast_rules)
        for scored, step in zip(scored_steps, scenario.steps, strict=True):
            if step.proposed.tool_name != "submit_eligibility_decision":
                continue
            if step.boundary_label is BoundaryLabel.POLICY_VIOLATION:
                assert scored.decision.verdict.value == "block", (
                    f"{scenario.scenario_id} expected submit-first to be BLOCKed"
                )
            elif step.boundary_label is BoundaryLabel.BENIGN:
                assert scored.decision.verdict.value == "allow", (
                    f"{scenario.scenario_id} expected verify-then-submit to be ALLOWed"
                )
