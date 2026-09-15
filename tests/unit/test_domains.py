from __future__ import annotations

from pathlib import Path

import pytest

from bossyk_sandbox.domains import domain_config
from bossyk_sandbox.env import bossyk_root
from bossyk_sandbox.instruments.base import ProposedAction, Verdict
from bossyk_sandbox.instruments.hardcoded_rule import RequireLookupBeforeCancel, RequirePassedCheck
from bossyk_sandbox.instruments.minimisation import (
    DEFAULT_REQUIRED_BAND_WIDTH_GBP,
    MinimisationInstrument,
)
from bossyk_sandbox.scenarios.loader import Scenario
from bossyk_sandbox.scenarios.runner import advice_fast_rules

# The policy YAMLs live in the private bossyk checkout (BOSSYK_ROOT), which
# exists on dev machines but not in CI — existence checks are gated on the
# checkout being present; path-shape assertions run everywhere.
requires_bossyk_checkout = pytest.mark.skipif(
    not bossyk_root().exists(),
    reason="needs the private bossyk checkout at BOSSYK_ROOT",
)


def test_airline_domain_scenarios_path_exists_on_disk() -> None:
    cfg = domain_config("airline")

    assert cfg.scenarios_path.exists()


@requires_bossyk_checkout
def test_airline_domain_policy_path_exists_on_disk() -> None:
    cfg = domain_config("airline")

    assert cfg.policy_path.exists()


def test_airline_domain_fast_rules_factory_returns_a_non_empty_list() -> None:
    cfg = domain_config("airline")

    fast_rules = cfg.fast_rules_factory()

    assert isinstance(fast_rules, list)
    assert len(fast_rules) > 0


def test_airline_domain_policy_path_basename_is_airline_support_v1_yaml() -> None:
    cfg = domain_config("airline")

    assert cfg.policy_path.name == "airline-support-v1.yaml"


def test_airline_domain_scenarios_path_ends_with_airline_scenarios_json() -> None:
    cfg = domain_config("airline")

    assert cfg.scenarios_path.as_posix().endswith("airline/scenarios.json")


def test_retail_domain_is_registered_with_expected_paths() -> None:
    cfg = domain_config("retail")

    assert cfg.policy_path.name == "retail-support-v1.yaml"
    assert cfg.scenarios_path.as_posix().endswith("retail/scenarios.json")
    assert cfg.scenarios_path.exists()


@requires_bossyk_checkout
def test_retail_domain_policy_path_exists_on_disk() -> None:
    cfg = domain_config("retail")

    assert cfg.policy_path.exists()


def test_retail_fast_rules_gate_the_order_write_tools() -> None:
    cfg = domain_config("retail")

    rules = cfg.fast_rules_factory()
    for rule in rules:
        assert isinstance(rule, RequireLookupBeforeCancel)
    gated = {rule.gated_tool for rule in rules if isinstance(rule, RequireLookupBeforeCancel)}

    assert gated == {
        "cancel_pending_order",
        "return_delivered_order_items",
        "modify_pending_order_payment",
        # SMACTR-derived in Phase 4 (see runner.retail_fast_rules).
        "modify_user_address",
    }


def test_domain_config_raises_key_error_for_unregistered_domain() -> None:
    with pytest.raises(KeyError):
        domain_config("telecom")


def test_outreach_domain_is_registered_with_expected_paths() -> None:
    # RED (slice 1): "outreach" is not yet in domains._DOMAIN_BUILDERS, so
    # this currently fails with KeyError('outreach').
    cfg = domain_config("outreach")

    assert cfg.policy_path.name == "outreach-outbound-v1.yaml"
    assert cfg.scenarios_path.as_posix().endswith("outreach/scenarios.json")
    assert cfg.scenarios_path.exists()


@requires_bossyk_checkout
def test_outreach_domain_policy_path_exists_on_disk() -> None:
    cfg = domain_config("outreach")

    assert cfg.policy_path.exists()


def test_outreach_fast_rules_gate_book_survey_and_place_call() -> None:
    # Test-Integrity note: updated for slice 2, P5 -- both outreach fast
    # rules are now the outcome-aware RequirePassedCheck, not slice 1's
    # precedence-only RequireLookupBeforeCancel (see
    # scenarios.runner.outreach_fast_rules).
    cfg = domain_config("outreach")

    rules = cfg.fast_rules_factory()
    for rule in rules:
        assert isinstance(rule, RequirePassedCheck)
    gated = {rule.gated_tool for rule in rules if isinstance(rule, RequirePassedCheck)}

    assert gated == {"book_survey", "place_call"}


def test_outreach_domain_policy_path_resolves_under_overridden_bossyk_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    custom_root = tmp_path / "custom-bossyk"
    monkeypatch.setenv("BOSSYK_ROOT", str(custom_root))

    cfg = domain_config("outreach")

    assert cfg.policy_path == custom_root / "data" / "policies" / "outreach-outbound-v1.yaml"


def test_airline_domain_policy_path_resolves_under_overridden_bossyk_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    custom_root = tmp_path / "custom-bossyk"
    monkeypatch.setenv("BOSSYK_ROOT", str(custom_root))

    cfg = domain_config("airline")

    assert cfg.policy_path == custom_root / "data" / "policies" / "airline-support-v1.yaml"


def test_retail_domain_policy_path_resolves_under_overridden_bossyk_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    custom_root = tmp_path / "custom-bossyk"
    monkeypatch.setenv("BOSSYK_ROOT", str(custom_root))

    cfg = domain_config("retail")

    assert cfg.policy_path == custom_root / "data" / "policies" / "retail-support-v1.yaml"


def test_advice_domain_is_registered_with_expected_paths() -> None:
    cfg = domain_config("advice")

    assert cfg.policy_path.name == "advice-support-v1.yaml"
    assert cfg.scenarios_path.as_posix().endswith("advice/scenarios.json")
    assert cfg.scenarios_path.exists()


@requires_bossyk_checkout
def test_advice_domain_policy_path_exists_on_disk() -> None:
    cfg = domain_config("advice")

    assert cfg.policy_path.exists()


# Test assumption CHANGED 2026-09-05, deliberately. This previously asserted
# `fast_rules == []` on the Phase-1 grounds that every advice tool is a read
# and the minimisation instrument was "Phase 2, out of scope". Phase 2 built
# that instrument and it is now wired, so the empty-list assumption is the
# thing being retired -- not a test that broke. The reads are still ungated
# (see the non-predicate guard below); what is gated is the free-threshold
# predicate `is_income_above`, whose repeated use reconstructs the value the
# banded tools exist to avoid disclosing.


def test_advice_fast_rules_factory_wires_the_minimisation_instrument() -> None:
    cfg = domain_config("advice")

    fast_rules = cfg.fast_rules_factory()

    assert len(fast_rules) == 1
    assert isinstance(fast_rules[0], MinimisationInstrument)


def test_advice_fast_rules_uses_the_scenarios_own_band_when_given_one() -> None:
    # advice-002 needs a 74_870 headroom band, not the 50_270 default.
    scenario = Scenario(
        scenario_id="advice-002", gated_tool=None, steps=[], required_band_width=74_870.0
    )

    rules = advice_fast_rules(scenario)

    instrument = rules[0]
    assert isinstance(instrument, MinimisationInstrument)
    assert instrument.config.required_band_width_gbp == 74_870.0


def test_advice_fast_rules_falls_back_to_the_default_band_without_a_scenario() -> None:
    instrument = advice_fast_rules()[0]

    assert isinstance(instrument, MinimisationInstrument)
    assert instrument.config.required_band_width_gbp == DEFAULT_REQUIRED_BAND_WIDTH_GBP


def test_advice_fast_rules_still_allows_the_domains_non_predicate_read_tools() -> None:
    # The blast-radius guard. Neither shipped advice scenario calls
    # `is_income_above`; they use these three. Wiring the instrument must not
    # change their outcome, or it would alter results it was never meant to.
    instrument = advice_fast_rules()[0]

    for tool in ("get_tax_position", "get_contribution_headroom", "get_customer_profile"):
        decision = instrument.score(ProposedAction(tool, {"ref": "ADV-0001"}), [])
        assert decision.verdict is Verdict.ALLOW, tool


def test_every_domain_fast_rules_factory_accepts_an_optional_scenario() -> None:
    # DomainConfig types the factory as taking an optional Scenario so the
    # advice band can be threaded per scenario; the other domains accept and
    # ignore it. Called both ways here so neither form can rot.
    scenario = Scenario(scenario_id="x", gated_tool=None, steps=[])

    for name in ("airline", "retail", "outreach", "advice", "advice-eligibility"):
        cfg = domain_config(name)
        assert isinstance(cfg.fast_rules_factory(), list), name
        assert isinstance(cfg.fast_rules_factory(scenario), list), name


def test_advice_domain_policy_path_resolves_under_overridden_bossyk_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    custom_root = tmp_path / "custom-bossyk"
    monkeypatch.setenv("BOSSYK_ROOT", str(custom_root))

    cfg = domain_config("advice")

    assert cfg.policy_path == custom_root / "data" / "policies" / "advice-support-v1.yaml"


# --- advice-eligibility (detector-training transfer domain; see the spec at
# coding-tasks/bossyk-sandbox/advice-eligibility-domain-spec.md) -------------


def test_advice_eligibility_domain_is_registered_with_expected_paths() -> None:
    cfg = domain_config("advice-eligibility")

    # Reuses the advice policy YAML (extend, don't fork) -- same prose judge,
    # +1 prohibited-action line for the eligibility-gating boundary.
    assert cfg.policy_path.name == "advice-support-v1.yaml"
    assert cfg.scenarios_path.as_posix().endswith("advice/eligibility-scenarios.json")
    assert cfg.scenarios_path.exists()


@requires_bossyk_checkout
def test_advice_eligibility_domain_policy_path_exists_on_disk() -> None:
    cfg = domain_config("advice-eligibility")

    assert cfg.policy_path.exists()


def test_advice_eligibility_fast_rules_gate_submit_on_a_prior_verify() -> None:
    # Test-integrity note (spec-parity audit,
    # detector-training-spec-parity-audit.md option (a)): this used to
    # assert a single-rule list. That assumption broke intentionally when
    # the domain was levelled up from 1 gated surface to 3 (contribution-
    # band revision + enrolment closure added alongside the original
    # submit_eligibility_decision), to match retail's structural surface
    # diversity. Updated to check all three rules explicitly, including the
    # 2-distinct-key_args property the audit called out.
    cfg = domain_config("advice-eligibility")

    rules = cfg.fast_rules_factory()
    require_lookup_rules = [rule for rule in rules if isinstance(rule, RequireLookupBeforeCancel)]

    assert len(rules) == 3
    assert len(require_lookup_rules) == 3  # every rule narrows -- none of another Instrument kind
    by_gated_tool = {rule.gated_tool: rule for rule in require_lookup_rules}
    assert set(by_gated_tool) == {
        "submit_eligibility_decision",
        "revise_contribution_band",
        "close_enrolment",
    }

    submit_rule = by_gated_tool["submit_eligibility_decision"]
    assert submit_rule.required_lookup_tool == "verify_eligibility"
    assert submit_rule.key_arg == "ref"

    revise_rule = by_gated_tool["revise_contribution_band"]
    assert revise_rule.required_lookup_tool == "verify_eligibility"
    assert revise_rule.key_arg == "ref"

    close_rule = by_gated_tool["close_enrolment"]
    assert close_rule.required_lookup_tool == "get_enrolment_status"
    assert close_rule.key_arg == "enrolment_id"

    distinct_key_args = {rule.key_arg for rule in require_lookup_rules}
    assert distinct_key_args == {"ref", "enrolment_id"}


def test_advice_eligibility_domain_policy_path_resolves_under_overridden_bossyk_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    custom_root = tmp_path / "custom-bossyk"
    monkeypatch.setenv("BOSSYK_ROOT", str(custom_root))

    cfg = domain_config("advice-eligibility")

    assert cfg.policy_path == custom_root / "data" / "policies" / "advice-support-v1.yaml"
