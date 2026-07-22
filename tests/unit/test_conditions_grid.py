from __future__ import annotations

from bossyk_sandbox.conditions.grid import AIRLINE_BOUNDARIES, AttackClass, ProbeCell, build_grid


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
