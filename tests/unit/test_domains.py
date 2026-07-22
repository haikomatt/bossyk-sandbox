from __future__ import annotations

import pytest

from bossyk_sandbox.domains import domain_config


def test_airline_domain_scenarios_path_exists_on_disk() -> None:
    cfg = domain_config("airline")

    assert cfg.scenarios_path.exists()


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


def test_retail_domain_is_registered_with_existing_paths() -> None:
    cfg = domain_config("retail")

    assert cfg.policy_path.name == "retail-support-v1.yaml"
    assert cfg.scenarios_path.as_posix().endswith("retail/scenarios.json")
    assert cfg.policy_path.exists()
    assert cfg.scenarios_path.exists()


def test_retail_fast_rules_gate_the_order_write_tools() -> None:
    cfg = domain_config("retail")

    gated = {rule.gated_tool for rule in cfg.fast_rules_factory()}

    assert gated == {
        "cancel_pending_order",
        "return_delivered_order_items",
        "modify_pending_order_payment",
    }


def test_domain_config_raises_key_error_for_unregistered_domain() -> None:
    with pytest.raises(KeyError):
        domain_config("telecom")
