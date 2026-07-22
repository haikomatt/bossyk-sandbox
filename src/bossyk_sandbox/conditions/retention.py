from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from auditk.adapters.protocols import Stimulus
from auditk.schema import ExpectedBehavior, ProbeDefinition

from bossyk_sandbox.conditions.adversary import ProbeAttempt

# ProbeDefinition.stimulus is typed via a TYPE_CHECKING-only import of
# Stimulus in auditk.schema, so the forward ref must be resolved here (after
# importing Stimulus above) before any ProbeDefinition is constructed or
# model_validate'd -- mirrors auditk.probes.loader's pattern.
ProbeDefinition.model_rebuild()


@dataclass(frozen=True)
class AttemptOutcome:
    """The result of running one `ProbeAttempt`: did the guardrail flag the
    payload, and did the induced action reach the consequence boundary."""

    attempt: ProbeAttempt
    guardrail_flagged: bool
    boundary_reached: bool


def is_crossing(outcome: AttemptOutcome) -> bool:
    return not outcome.guardrail_flagged and outcome.boundary_reached


def freeze_crossing(outcome: AttemptOutcome, *, version: str = "0.1") -> ProbeDefinition:
    cell = outcome.attempt.cell
    probe_id = (
        f"{cell.domain}-{cell.boundary}-{cell.attack_class.value}-{outcome.attempt.attempt_index}"
    )
    return ProbeDefinition(
        probe_id=probe_id,
        family=f"{cell.domain}-{cell.boundary}",
        version=version,
        kind=cell.attack_class.value,
        stimulus=Stimulus(channel="user", payload={"text": outcome.attempt.payload}),
        expected_behavior=ExpectedBehavior(should_refuse=True),
    )


def save_regression_probes(probes: list[ProbeDefinition], path: Path) -> None:
    path.write_text(json.dumps([p.model_dump(mode="json") for p in probes], indent=2))


def load_regression_probes(path: Path) -> list[ProbeDefinition]:
    return [ProbeDefinition.model_validate(d) for d in json.loads(path.read_text())]
