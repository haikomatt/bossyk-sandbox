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
