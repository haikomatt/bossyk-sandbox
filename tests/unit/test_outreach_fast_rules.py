"""Tests for `outreach_fast_rules()` (bossyk-sandbox slice 1 -> slice 2, P5).

Test-Integrity note (pm-workflow.md): this file's slice-1 assumption --
"every outreach fast rule is `RequireLookupBeforeCancel`" -- is exactly what
slice 2, P5 is designed to invalidate (that's the whole point of adding the
outcome-aware `RequirePassedCheck`). The code was not wrong; the test's
assumption was scoped to slice 1 only. Updated here, not silently: both
outreach fast rules are now `RequirePassedCheck` --

- `book_survey` <- `check_eligibility` (boundary 3, upgraded from
  precedence-only to outcome-aware, per outreach-domain-cleanroom-spec.md
  boundary 3).
- `place_call` <- `check_suppression` (boundary 1, newly enactable this
  slice -- precedence gating could never express it).
"""

from __future__ import annotations

from bossyk_sandbox.instruments.hardcoded_rule import RequirePassedCheck
from bossyk_sandbox.scenarios.runner import outreach_fast_rules


def test_outreach_fast_rules_are_both_outcome_aware() -> None:
    rules = outreach_fast_rules()

    assert len(rules) == 2
    assert all(isinstance(rule, RequirePassedCheck) for rule in rules)
    gated = {rule.gated_tool for rule in rules if isinstance(rule, RequirePassedCheck)}
    assert gated == {"book_survey", "place_call"}


def test_outreach_fast_rules_book_survey_rule_is_keyed_on_prospect_id() -> None:
    rules = outreach_fast_rules()
    book_survey_rules = [
        rule
        for rule in rules
        if isinstance(rule, RequirePassedCheck) and rule.gated_tool == "book_survey"
    ]

    assert len(book_survey_rules) == 1
    book_survey_rule = book_survey_rules[0]
    assert book_survey_rule.check_tool == "check_eligibility"
    assert book_survey_rule.key_arg == "prospect_id"


def test_outreach_fast_rules_place_call_rule_is_keyed_on_phone() -> None:
    rules = outreach_fast_rules()
    place_call_rules = [
        rule
        for rule in rules
        if isinstance(rule, RequirePassedCheck) and rule.gated_tool == "place_call"
    ]

    assert len(place_call_rules) == 1
    place_call_rule = place_call_rules[0]
    assert place_call_rule.check_tool == "check_suppression"
    assert place_call_rule.key_arg == "phone"


def test_outreach_fast_rules_book_survey_predicate_requires_eligible_true() -> None:
    rules = outreach_fast_rules()
    book_survey_rule = next(
        rule
        for rule in rules
        if isinstance(rule, RequirePassedCheck) and rule.gated_tool == "book_survey"
    )

    assert book_survey_rule.predicate({"eligible": True}) is True
    assert book_survey_rule.predicate({"eligible": False}) is False


def test_outreach_fast_rules_place_call_predicate_requires_a_clean_suppression_check() -> None:
    rules = outreach_fast_rules()
    place_call_rule = next(
        rule
        for rule in rules
        if isinstance(rule, RequirePassedCheck) and rule.gated_tool == "place_call"
    )

    assert place_call_rule.predicate({"on_tps": False, "opted_out": False}) is True
    assert place_call_rule.predicate({"on_tps": True, "opted_out": False}) is False
    assert place_call_rule.predicate({"on_tps": False, "opted_out": True}) is False
