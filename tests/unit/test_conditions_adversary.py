from __future__ import annotations

from bossyk_sandbox.conditions.adversary import AdversaryIntensity, StubAdversary, budget_for
from bossyk_sandbox.conditions.grid import AttackClass, ProbeCell


def test_budget_for_gentle_is_two() -> None:
    assert budget_for(AdversaryIntensity.GENTLE) == 2


def test_budget_for_taxing_is_five() -> None:
    assert budget_for(AdversaryIntensity.TAXING) == 5


def test_budget_for_aggressive_is_ten() -> None:
    assert budget_for(AdversaryIntensity.AGGRESSIVE) == 10


def test_stub_adversary_generates_exactly_budget_attempts_with_sequential_index() -> None:
    cell = ProbeCell("airline", AttackClass.JAILBREAK, "cancel_without_lookup")
    adversary = StubAdversary(payloads_by_class={AttackClass.JAILBREAK: ["payload-a", "payload-b"]})

    attempts = adversary.generate_attempts(cell, budget=5)

    assert len(attempts) == 5
    assert [attempt.attempt_index for attempt in attempts] == [0, 1, 2, 3, 4]


def test_stub_adversary_payloads_correspond_to_the_cells_attack_class() -> None:
    cell = ProbeCell("airline", AttackClass.PII_LEAK, "pii_disclosure")
    scripted = ["leak-payload-1", "leak-payload-2"]
    adversary = StubAdversary(payloads_by_class={AttackClass.PII_LEAK: scripted})

    attempts = adversary.generate_attempts(cell, budget=2)

    assert all(attempt.payload in scripted for attempt in attempts)
