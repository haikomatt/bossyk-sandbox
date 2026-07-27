from __future__ import annotations

from bossyk_sandbox.conditions.adversary import ProbeAttempt, TokenUsage
from bossyk_sandbox.conditions.grid import AttackClass, ProbeCell
from bossyk_sandbox.scoring.cost import (
    JudgeCallRecord,
    build_judge_token_ledger,
    build_token_ledger,
)

# The token ledger answers "what did each adversary model actually cost to
# run?" (plan §15A) -- it aggregates per-attempt token usage per model id so a
# variance run (fireworks-deepseek vs anthropic-fable) reports each attacker's
# real input/output token spend, including tokens burned on refused attempts.

CELL = ProbeCell("airline", AttackClass.JAILBREAK, "cancel_without_lookup")


def _attempt(model: str, *, refused: bool = False, usage: TokenUsage | None = None) -> ProbeAttempt:
    return ProbeAttempt(
        cell=CELL,
        payload="" if refused else "payload",
        attempt_index=0,
        metadata={"model": model},
        status="refused" if refused else "ok",
        usage=usage if usage is not None else TokenUsage(),
    )


def test_build_token_ledger_aggregates_usage_per_model() -> None:
    attempts = [
        _attempt("m1", usage=TokenUsage(10, 20)),
        _attempt("m1", usage=TokenUsage(5, 5)),
        _attempt("m2", usage=TokenUsage(1, 2)),
    ]

    ledger = build_token_ledger(attempts)

    assert set(ledger) == {"m1", "m2"}
    assert ledger["m1"].calls == 2
    assert ledger["m1"].usage == TokenUsage(15, 25)
    assert ledger["m1"].usage.total_tokens == 40
    assert ledger["m2"].calls == 1
    assert ledger["m2"].usage == TokenUsage(1, 2)


def test_build_token_ledger_counts_refused_calls_but_still_bills_their_tokens() -> None:
    attempts = [
        _attempt("m1", refused=True, usage=TokenUsage(50, 4)),
        _attempt("m1", usage=TokenUsage(10, 20)),
    ]

    ledger = build_token_ledger(attempts)

    assert ledger["m1"].calls == 2
    assert ledger["m1"].refused_calls == 1
    assert ledger["m1"].usage == TokenUsage(60, 24)


def test_build_token_ledger_labels_attempts_without_a_model_as_unknown() -> None:
    attempts = [ProbeAttempt(cell=CELL, payload="p", attempt_index=0)]

    ledger = build_token_ledger(attempts)

    assert "unknown" in ledger
    assert ledger["unknown"].calls == 1


def test_build_token_ledger_of_no_attempts_is_empty() -> None:
    assert build_token_ledger([]) == {}


# --- judge-call token ledger (live-h2h4 L1 #6, plan §15B) --------------------


def test_build_judge_token_ledger_aggregates_usage_per_instrument() -> None:
    records = [
        JudgeCallRecord(instrument="policy", usage=TokenUsage(10, 20)),
        JudgeCallRecord(instrument="policy", usage=TokenUsage(5, 5)),
        JudgeCallRecord(instrument="drift", usage=TokenUsage(1, 2)),
    ]

    ledger = build_judge_token_ledger(records)

    assert set(ledger) == {"policy", "drift"}
    assert ledger["policy"].calls == 2
    assert ledger["policy"].usage == TokenUsage(15, 25)
    assert ledger["policy"].error_calls == 0
    assert ledger["drift"].calls == 1
    assert ledger["drift"].usage == TokenUsage(1, 2)


def test_build_judge_token_ledger_counts_errored_calls_with_their_zero_usage() -> None:
    records = [
        JudgeCallRecord(instrument="policy", usage=TokenUsage(), errored=True),
        JudgeCallRecord(instrument="policy", usage=TokenUsage(10, 20)),
    ]

    ledger = build_judge_token_ledger(records)

    assert ledger["policy"].calls == 2
    assert ledger["policy"].error_calls == 1
    assert ledger["policy"].usage == TokenUsage(10, 20)


def test_build_judge_token_ledger_of_no_records_is_empty() -> None:
    assert build_judge_token_ledger([]) == {}
