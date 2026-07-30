from __future__ import annotations

from pathlib import Path

import pytest

from bossyk_sandbox.domains import domain_config
from bossyk_sandbox.env import bossyk_root
from bossyk_sandbox.instruments.hardcoded_rule import RequireLookupBeforeCancel, RequirePassedCheck

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
