from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from bossyk_sandbox.conditions.retention import AttemptOutcome

# GREEN intent: BypassRateResult.wilson_ci95 reuses the existing
# `bossyk_sandbox.scoring.orthogonality.wilson_interval` Wilson-score-interval
# helper rather than adding a parallel CI implementation. Not imported yet --
# this is a RED skeleton.


@dataclass(frozen=True)
class BypassRateResult:
    n_attempts: int
    n_bypassed: int

    def rate(self) -> float:
        raise NotImplementedError

    def wilson_ci95(self) -> tuple[float, float]:
        # GREEN intent: wilson_interval(self.n_bypassed, self.n_attempts)
        raise NotImplementedError


def bypass_rate(outcomes: list[AttemptOutcome]) -> BypassRateResult:
    # GREEN intent: bypassed = not guardrail_flagged
    raise NotImplementedError


def bypass_rate_by(
    outcomes: list[AttemptOutcome], key: Callable[[AttemptOutcome], str]
) -> dict[str, BypassRateResult]:
    raise NotImplementedError
