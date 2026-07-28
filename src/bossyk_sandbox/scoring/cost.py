from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from bossyk_sandbox.conditions.adversary import ProbeAttempt, TokenUsage

UNKNOWN_MODEL = "unknown"


@dataclass(frozen=True)
class ModelLedgerEntry:
    """Per-model roll-up of adversary calls and the tokens they cost -- one
    row of the token ledger (plan §15A). `refused_calls` is a subset of
    `calls`; `usage` bills refused and non-refused attempts alike."""

    model: str
    calls: int
    refused_calls: int
    usage: TokenUsage


def build_token_ledger(attempts: Iterable[ProbeAttempt]) -> dict[str, ModelLedgerEntry]:
    """Aggregate per-attempt token usage into one entry per adversary model
    id (read from `attempt.metadata['model']`, falling back to 'unknown').
    Counts total and refused calls and sums input/output tokens across all
    attempts."""
    calls: dict[str, int] = {}
    refused: dict[str, int] = {}
    usage: dict[str, TokenUsage] = {}
    for attempt in attempts:
        model = attempt.metadata.get("model", UNKNOWN_MODEL)
        calls[model] = calls.get(model, 0) + 1
        refused[model] = refused.get(model, 0) + (1 if attempt.refused else 0)
        usage[model] = usage.get(model, TokenUsage()) + attempt.usage
    return {
        model: ModelLedgerEntry(
            model=model,
            calls=calls[model],
            refused_calls=refused[model],
            usage=usage[model],
        )
        for model in calls
    }


@dataclass(frozen=True)
class JudgeCallRecord:
    """One judge call's token usage -- the live-H2/H4 counterpart of
    `ProbeAttempt` for this ledger: policy/drift judge calls cost tokens
    too (plan §15B: "extend this session's scoring/cost.py token ledger to
    the judge calls"). `errored` calls carry empty `usage` (Finding 4: an
    error is unavailable, not a zero-cost success -- but a failed judge
    call's tokens aren't independently observable from `PolicyStepResult`,
    so error-path usage is honestly zero, not fabricated)."""

    instrument: str
    usage: TokenUsage
    errored: bool = False


@dataclass(frozen=True)
class JudgeLedgerEntry:
    """Per-instrument roll-up of judge calls and the tokens they cost --
    one row of the judge-call ledger, parallel to `ModelLedgerEntry`."""

    instrument: str
    calls: int
    error_calls: int
    usage: TokenUsage


def build_judge_token_ledger(records: Iterable[JudgeCallRecord]) -> dict[str, JudgeLedgerEntry]:
    """Aggregate per-call token usage into one entry per judge instrument
    name (e.g. `"policy"`). Counts total and errored calls and sums
    input/output tokens across all calls, mirroring `build_token_ledger`'s
    per-model aggregation."""
    calls: dict[str, int] = {}
    errors: dict[str, int] = {}
    usage: dict[str, TokenUsage] = {}
    for record in records:
        calls[record.instrument] = calls.get(record.instrument, 0) + 1
        errors[record.instrument] = errors.get(record.instrument, 0) + (1 if record.errored else 0)
        usage[record.instrument] = usage.get(record.instrument, TokenUsage()) + record.usage
    return {
        instrument: JudgeLedgerEntry(
            instrument=instrument,
            calls=calls[instrument],
            error_calls=errors[instrument],
            usage=usage[instrument],
        )
        for instrument in calls
    }
