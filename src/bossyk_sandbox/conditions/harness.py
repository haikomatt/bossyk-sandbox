from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from auditk.schema import ProbeDefinition

from bossyk_sandbox.conditions.adversary import Adversary, ProbeAttempt
from bossyk_sandbox.conditions.grid import ProbeCell
from bossyk_sandbox.conditions.retention import AttemptOutcome, freeze_crossing, is_crossing
from bossyk_sandbox.guardrail.guardrail import Guardrail
from bossyk_sandbox.scoring.h1 import BypassRateResult, bypass_rate, bypass_rate_by


def _all_boundaries_reached(attempt: ProbeAttempt) -> bool:
    # Phase 2a scripted-crossing model (SCOUT §E): adversary-generated
    # attempts are real attacks by construction, so boundary_reached is True
    # by default. Phase 3's live-agent oracle replaces this -- hence it's an
    # injectable parameter, not hardcoded.
    return True


@dataclass(frozen=True)
class ProbeGridRun:
    """The full result of running an adversary against every cell of a
    probe grid through a guardrail: raw outcomes, the crossings distilled
    out of them, the regression probes frozen from those crossings, and the
    H1 bypass-rate rollups (overall / by attack class / by boundary)."""

    outcomes: list[AttemptOutcome]
    crossings: list[AttemptOutcome]
    regression_probes: list[ProbeDefinition]
    h1_overall: BypassRateResult
    h1_by_class: dict[str, BypassRateResult]
    h1_by_boundary: dict[str, BypassRateResult]


def run_probe_grid(
    cells: list[ProbeCell],
    adversary: Adversary,
    guardrail: Guardrail,
    budget: int,
    boundary_oracle: Callable[[ProbeAttempt], bool] = _all_boundaries_reached,
) -> ProbeGridRun:
    outcomes: list[AttemptOutcome] = []
    for cell in cells:
        for attempt in adversary.generate_attempts(cell, budget):
            if attempt.status != "ok":
                # Either the adversary model declined to generate a payload
                # (status="refused") or the provider call itself failed
                # (status="error", e.g. an empty/malformed response) -- in
                # both cases there is no attack to inspect, so the guardrail
                # is never consulted and no boundary can be reached. Recorded
                # (so H1 can report the refusal/error rate and drop it from
                # the bypass denominator); never a crossing, never frozen.
                outcomes.append(
                    AttemptOutcome(attempt=attempt, guardrail_flagged=False, boundary_reached=False)
                )
                continue
            verdict = guardrail.inspect(attempt.payload)
            reached = boundary_oracle(attempt)
            outcomes.append(
                AttemptOutcome(
                    attempt=attempt,
                    guardrail_flagged=verdict.flagged,
                    boundary_reached=reached,
                )
            )

    crossings = [outcome for outcome in outcomes if is_crossing(outcome)]
    regression_probes = [freeze_crossing(outcome) for outcome in crossings]
    return ProbeGridRun(
        outcomes=outcomes,
        crossings=crossings,
        regression_probes=regression_probes,
        h1_overall=bypass_rate(outcomes),
        h1_by_class=bypass_rate_by(
            outcomes, key=lambda outcome: outcome.attempt.cell.attack_class.value
        ),
        h1_by_boundary=bypass_rate_by(outcomes, key=lambda outcome: outcome.attempt.cell.boundary),
    )
