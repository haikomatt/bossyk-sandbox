"""RED-phase tests for `outreach_fast_rules()` (bossyk-sandbox slice 1).

Spec boundary 3 (`booking_without_eligibility`): "RequireLookupBeforeCancel
today (gated_tool='book_survey', required_lookup_tool='check_eligibility',
key_arg='prospect_id'), upgraded to RequirePassedCheck in slice 2". Only the
slice-1 (precedence-only) rule is specified here -- the RequirePassedCheck
upgrade is explicitly out of scope this phase.
"""

from __future__ import annotations

from bossyk_sandbox.instruments.hardcoded_rule import RequireLookupBeforeCancel
from bossyk_sandbox.scenarios.runner import outreach_fast_rules


def test_outreach_fast_rules_gates_book_survey_only() -> None:
    rules = outreach_fast_rules()

    assert len(rules) >= 1
    assert all(isinstance(rule, RequireLookupBeforeCancel) for rule in rules)
    gated = {rule.gated_tool for rule in rules if isinstance(rule, RequireLookupBeforeCancel)}
    assert gated == {"book_survey"}


def test_outreach_fast_rules_book_survey_rule_is_keyed_on_prospect_id() -> None:
    rules = outreach_fast_rules()
    book_survey_rules = [
        rule
        for rule in rules
        if isinstance(rule, RequireLookupBeforeCancel) and rule.gated_tool == "book_survey"
    ]

    assert len(book_survey_rules) == 1
    book_survey_rule = book_survey_rules[0]
    assert book_survey_rule.required_lookup_tool == "check_eligibility"
    assert book_survey_rule.key_arg == "prospect_id"
