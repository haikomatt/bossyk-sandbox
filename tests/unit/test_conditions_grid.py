from __future__ import annotations

import pytest

from bossyk_sandbox.conditions.grid import (
    AIRLINE_BOUNDARIES,
    RETAIL_BOUNDARIES,
    AttackClass,
    ProbeCell,
    boundaries_for,
    build_grid,
)


def test_grid_length_is_the_cartesian_product_of_classes_and_boundaries() -> None:
    classes = [AttackClass.PROMPT_INJECTION, AttackClass.JAILBREAK]
    boundaries = AIRLINE_BOUNDARIES

    grid = build_grid("airline", classes, boundaries)

    assert len(grid) == len(classes) * len(boundaries)


def test_grid_contains_the_expected_probe_cell() -> None:
    classes = [AttackClass.PII_LEAK]
    boundaries = ["pii_disclosure"]

    grid = build_grid("airline", classes, boundaries)

    assert (
        ProbeCell(domain="airline", attack_class=AttackClass.PII_LEAK, boundary="pii_disclosure")
        in grid
    )


def test_grid_order_is_deterministic_classes_outer_boundaries_inner() -> None:
    classes = [AttackClass.JAILBREAK, AttackClass.TOOL_MISUSE]
    boundaries = ["cancel_without_lookup", "refund_over_threshold"]

    grid = build_grid("airline", classes, boundaries)

    assert grid == [
        ProbeCell("airline", AttackClass.JAILBREAK, "cancel_without_lookup"),
        ProbeCell("airline", AttackClass.JAILBREAK, "refund_over_threshold"),
        ProbeCell("airline", AttackClass.TOOL_MISUSE, "cancel_without_lookup"),
        ProbeCell("airline", AttackClass.TOOL_MISUSE, "refund_over_threshold"),
    ]


def test_retail_boundaries_are_the_four_expected_names() -> None:
    assert RETAIL_BOUNDARIES == [
        "cancel_without_auth",
        "refund_over_threshold",
        "pii_disclosure",
        "unauthorized_modification",
    ]


def test_boundaries_for_returns_the_retail_axis() -> None:
    assert boundaries_for("retail") == RETAIL_BOUNDARIES


def test_boundaries_for_returns_the_airline_axis() -> None:
    assert boundaries_for("airline") == AIRLINE_BOUNDARIES


def test_boundaries_for_unknown_domain_raises() -> None:
    with pytest.raises(KeyError):
        boundaries_for("telecom")


# --- outreach domain (bossyk-sandbox slice 1 RED) ----------------------------
# RED: "outreach" is not yet registered in BOUNDARIES_BY_DOMAIN, so this
# currently fails with KeyError('outreach'). The six names come from
# outreach-domain-cleanroom-spec.md's "six boundaries" table (D6).


def test_boundaries_for_outreach_returns_the_six_named_boundaries() -> None:
    assert boundaries_for("outreach") == [
        "contact_without_consent",
        "out_of_hours_contact",
        "booking_without_eligibility",
        "unauthorised_incentive",
        "prohibited_financial_promotion",
        "pii_cross_contact",
    ]
