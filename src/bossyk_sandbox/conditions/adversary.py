from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal, Protocol

from bossyk_sandbox.conditions.grid import AttackClass, ProbeCell


class AdversaryIntensity(StrEnum):
    """Configured attack pressure per probe cell (SCOUT.md §E)."""

    GENTLE = "gentle"
    TAXING = "taxing"
    AGGRESSIVE = "aggressive"


INTENSITY_BUDGET: dict[AdversaryIntensity, int] = {
    AdversaryIntensity.GENTLE: 2,
    AdversaryIntensity.TAXING: 5,
    AdversaryIntensity.AGGRESSIVE: 10,
}


def budget_for(intensity: AdversaryIntensity) -> int:
    return INTENSITY_BUDGET[intensity]


@dataclass(frozen=True)
class TokenUsage:
    """Input/output token counts for one or many model calls -- the unit
    of the per-model cost ledger (scoring/cost.py). Frozen and additive so
    usage can be summed across a whole run."""

    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )


def usage_from_anthropic(response: Any) -> TokenUsage:
    """Read token usage off an Anthropic Messages response
    (`response.usage.input_tokens` / `.output_tokens`). Duck-typed (no SDK
    import) and tolerant of a missing `usage` -- returns empty usage rather
    than raising. Reads only `.usage`, never `.content`, so it is safe to
    call before branching on a refusal."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return TokenUsage()
    return TokenUsage(
        input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
        output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
    )


def usage_from_langchain(message: Any) -> TokenUsage:
    """Read token usage off a langchain chat message (`message.usage_metadata`,
    a dict with `input_tokens`/`output_tokens`). Duck-typed and tolerant of a
    provider that returns no usage (`usage_metadata` is None/empty)."""
    metadata = getattr(message, "usage_metadata", None)
    if not metadata:
        return TokenUsage()
    return TokenUsage(
        input_tokens=int(metadata.get("input_tokens", 0) or 0),
        output_tokens=int(metadata.get("output_tokens", 0) or 0),
    )


@dataclass(frozen=True)
class ChatResult:
    """One `ChatClient.complete` result -- lets a client signal that the
    underlying model declined to produce a payload (some frontier models
    refuse red-team generation requests) instead of conflating a refusal
    with an empty or garbage payload.

    `status` (Finding 9) is the source of truth; `refused` is a read-only
    property derived from `status == "refused"`."""

    text: str
    status: Literal["ok", "refused", "error"] = "ok"
    detail: str = ""
    usage: TokenUsage = field(default_factory=TokenUsage)

    @property
    def refused(self) -> bool:
        return self.status == "refused"


@dataclass(frozen=True)
class ProbeAttempt:
    """One adversary attempt at a `ProbeCell` -- the injected untrusted
    content / jailbreak phrasing plus its position in the per-cell budget."""

    cell: ProbeCell
    payload: str
    attempt_index: int
    # Any: adversary-specific free-form bookkeeping (e.g. mutation lineage,
    # real-adversary model call id) -- shape is intentionally not fixed by
    # this contract, mirrors auditk's ProbeDefinition.metadata pattern.
    metadata: dict[str, Any] = field(default_factory=dict)
    status: Literal["ok", "refused", "error"] = "ok"
    usage: TokenUsage = field(default_factory=TokenUsage)

    @property
    def refused(self) -> bool:
        return self.status == "refused"


def validate_payload(result: ChatResult) -> ChatResult:
    """Guards against Finding 9: an `"ok"` result with empty (or
    whitespace-only) text is not a genuine successful call -- it is an
    empty-payload failure mode (e.g. a non-refusal Anthropic response with
    no text block) that must not be counted as a real attack payload
    downstream. Returns a copy with `status="error"` in that case;
    otherwise returns `result` unchanged, including a `"refused"` result
    with empty text, which is a legitimate (not erroneous) empty payload."""
    if result.status == "ok" and not result.text.strip():
        return dataclasses.replace(result, status="error", detail="empty model response")
    return result


class Adversary(Protocol):
    """Generates attempts against a cell, following the
    `DriftScorer`/`PolicyJudgeClient` stub/real split pattern."""

    def generate_attempts(self, cell: ProbeCell, budget: int) -> list[ProbeAttempt]: ...


@dataclass
class StubAdversary:
    """Fixed scripted payloads per attack class -- deterministic, no model
    call, for the unit-test suite."""

    payloads_by_class: dict[AttackClass, list[str]]

    def generate_attempts(self, cell: ProbeCell, budget: int) -> list[ProbeAttempt]:
        # Return exactly `budget` attempts, cycling the scripted payloads for
        # `cell.attack_class`, attempt_index 0..budget-1. metadata["model"] is
        # "stub" (matching h1_bench's smoke label) so the token ledger keys
        # these under "stub" rather than its "unknown" fallback.
        payloads = self.payloads_by_class[cell.attack_class]
        return [
            ProbeAttempt(
                cell=cell,
                payload=payloads[i % len(payloads)],
                attempt_index=i,
                metadata={"model": "stub"},
            )
            for i in range(budget)
        ]
