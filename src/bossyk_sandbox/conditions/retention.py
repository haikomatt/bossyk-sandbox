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


def freeze_attempt(attempt: ProbeAttempt, *, version: str = "0.1") -> ProbeDefinition:
    """Freeze a raw adversary `ProbeAttempt` into a `ProbeDefinition` for a
    live corpus -- the grounded-attack counterpart of `freeze_crossing`,
    which additionally needs an `AttemptOutcome` (guardrail flag + boundary
    reach). A grounded attack is a candidate live input, not yet a guardrail
    crossing, so it carries no outcome; the frozen probe is the same shape
    `replay_crossing` and the live bench consume (family = '<domain>-<boundary>')."""
    cell = attempt.cell
    probe_id = f"{cell.domain}-{cell.boundary}-{cell.attack_class.value}-{attempt.attempt_index}"
    return ProbeDefinition(
        probe_id=probe_id,
        family=f"{cell.domain}-{cell.boundary}",
        version=version,
        kind=cell.attack_class.value,
        stimulus=Stimulus(channel="user", payload={"text": attempt.payload}),
        expected_behavior=ExpectedBehavior(should_refuse=True),
    )


def freeze_crossing(outcome: AttemptOutcome, *, version: str = "0.1") -> ProbeDefinition:
    """A crossing's frozen probe is exactly `freeze_attempt` on its attempt
    -- the guardrail-flag/boundary-reach predicates that define a crossing
    (`is_crossing`) don't change the probe's shape."""
    return freeze_attempt(outcome.attempt, version=version)


def save_regression_probes(probes: list[ProbeDefinition], path: Path) -> None:
    path.write_text(json.dumps([p.model_dump(mode="json") for p in probes], indent=2))


def load_regression_probes(path: Path) -> list[ProbeDefinition]:
    return [ProbeDefinition.model_validate(d) for d in json.loads(path.read_text())]


def append_regression_probe(probe: ProbeDefinition, path: Path) -> None:
    """Freezes `probe` into the regression-probe file at `path`, creating it
    if missing. Raises `ValueError` (naming the duplicate `probe_id`) instead
    of silently overwriting when the probe is already frozen there -- the
    file is left untouched in that case."""
    probes = load_regression_probes(path) if path.exists() else []
    if any(existing.probe_id == probe.probe_id for existing in probes):
        raise ValueError(f"probe_id {probe.probe_id!r} is already frozen at {path}")
    probes.append(probe)
    save_regression_probes(probes, path)
