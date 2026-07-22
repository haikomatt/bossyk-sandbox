from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from auditk.schema import ProbeDefinition

from bossyk_sandbox.conditions.adversary import ProbeAttempt


@dataclass(frozen=True)
class AttemptOutcome:
    """The result of running one `ProbeAttempt`: did the guardrail flag the
    payload, and did the induced action reach the consequence boundary."""

    attempt: ProbeAttempt
    guardrail_flagged: bool
    boundary_reached: bool


def is_crossing(outcome: AttemptOutcome) -> bool:
    # GREEN intent: not guardrail_flagged and boundary_reached
    raise NotImplementedError


def freeze_crossing(outcome: AttemptOutcome, *, version: str = "0.1") -> ProbeDefinition:
    # GREEN intent: build an auditk ProbeDefinition with
    # kind=outcome.attempt.cell.attack_class.value,
    # family=f"{outcome.attempt.cell.domain}-{outcome.attempt.cell.boundary}",
    # stimulus carrying the payload, expected_behavior = refuse/hold.
    raise NotImplementedError


def save_regression_probes(probes: list[ProbeDefinition], path: Path) -> None:
    raise NotImplementedError


def load_regression_probes(path: Path) -> list[ProbeDefinition]:
    raise NotImplementedError
