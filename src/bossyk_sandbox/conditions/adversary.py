from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol

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
        # `cell.attack_class`, attempt_index 0..budget-1.
        payloads = self.payloads_by_class[cell.attack_class]
        return [
            ProbeAttempt(cell=cell, payload=payloads[i % len(payloads)], attempt_index=i)
            for i in range(budget)
        ]
