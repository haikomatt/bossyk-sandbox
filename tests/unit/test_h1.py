from __future__ import annotations

from bossyk_sandbox.conditions.adversary import ProbeAttempt
from bossyk_sandbox.conditions.grid import AttackClass, ProbeCell
from bossyk_sandbox.conditions.retention import AttemptOutcome
from bossyk_sandbox.scoring.h1 import bypass_rate, bypass_rate_by


def _outcome(attack_class: AttackClass, *, guardrail_flagged: bool) -> AttemptOutcome:
    cell = ProbeCell("airline", attack_class, "cancel_without_lookup")
    attempt = ProbeAttempt(cell=cell, payload="payload", attempt_index=0)
    return AttemptOutcome(
        attempt=attempt, guardrail_flagged=guardrail_flagged, boundary_reached=True
    )


def test_bypass_rate_counts_unflagged_attempts_as_bypassed() -> None:
    outcomes = [
        _outcome(AttackClass.JAILBREAK, guardrail_flagged=False),
        _outcome(AttackClass.JAILBREAK, guardrail_flagged=False),
        _outcome(AttackClass.JAILBREAK, guardrail_flagged=True),
        _outcome(AttackClass.JAILBREAK, guardrail_flagged=True),
    ]

    result = bypass_rate(outcomes)

    assert result.n_attempts == 4
    assert result.n_bypassed == 2
    assert result.rate() == 0.5


def test_bypass_rate_by_groups_by_key_function() -> None:
    outcomes = [
        _outcome(AttackClass.JAILBREAK, guardrail_flagged=False),
        _outcome(AttackClass.PROMPT_INJECTION, guardrail_flagged=True),
    ]

    grouped = bypass_rate_by(outcomes, key=lambda outcome: outcome.attempt.cell.attack_class.value)

    assert grouped["jailbreak"].n_attempts == 1
    assert grouped["prompt_injection"].n_attempts == 1


def test_wilson_ci95_brackets_the_point_estimate() -> None:
    outcomes = [
        _outcome(AttackClass.JAILBREAK, guardrail_flagged=False),
        _outcome(AttackClass.JAILBREAK, guardrail_flagged=False),
        _outcome(AttackClass.JAILBREAK, guardrail_flagged=False),
        _outcome(AttackClass.JAILBREAK, guardrail_flagged=True),
    ]

    result = bypass_rate(outcomes)
    low, high = result.wilson_ci95()

    assert low <= result.rate() <= high
