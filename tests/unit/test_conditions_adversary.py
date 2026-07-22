from __future__ import annotations

from bossyk_sandbox.conditions.adversary import (
    AdversaryIntensity,
    ChatResult,
    ProbeAttempt,
    StubAdversary,
    TokenUsage,
    budget_for,
)
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


def test_stub_adversary_labels_its_model_as_stub() -> None:
    # So the smoke-mode token ledger keys stub attempts under "stub", matching
    # the h1_bench smoke label, rather than the "unknown" fallback.
    cell = ProbeCell("airline", AttackClass.JAILBREAK, "cancel_without_lookup")
    adversary = StubAdversary(payloads_by_class={AttackClass.JAILBREAK: ["p"]})

    attempts = adversary.generate_attempts(cell, budget=2)

    assert all(attempt.metadata["model"] == "stub" for attempt in attempts)


# --- token usage (per-model cost ledger, plan §15A) -------------------------


def test_token_usage_defaults_to_zero_and_reports_total() -> None:
    empty = TokenUsage()
    assert (empty.input_tokens, empty.output_tokens, empty.total_tokens) == (0, 0, 0)
    assert TokenUsage(input_tokens=10, output_tokens=25).total_tokens == 35


def test_token_usage_adds_componentwise() -> None:
    # Aggregating usage across many calls must sum input and output separately.
    assert TokenUsage(3, 4) + TokenUsage(10, 20) == TokenUsage(13, 24)


def test_chat_result_carries_empty_usage_by_default() -> None:
    assert ChatResult(text="payload").usage == TokenUsage()


def test_probe_attempt_carries_empty_usage_by_default() -> None:
    cell = ProbeCell("airline", AttackClass.JAILBREAK, "cancel_without_lookup")
    assert ProbeAttempt(cell=cell, payload="p", attempt_index=0).usage == TokenUsage()
