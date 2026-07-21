from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

Z_95 = 1.959963984540054


@dataclass(frozen=True)
class StepMembership:
    """One scored step's membership in the three orthogonality axes."""

    step_id: str
    drift_fires: bool
    policy_fires: bool
    outcome_violation: bool


@dataclass(frozen=True)
class MembershipCell:
    drift_fires: bool
    policy_fires: bool
    outcome_violation: bool


@dataclass(frozen=True)
class CellCount:
    cell: MembershipCell
    count: int
    proportion: float
    ci_low: float
    ci_high: float


def wilson_interval(successes: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion. Returns (0.0, 0.0)
    for n == 0 rather than raising — an empty sample has no defined rate."""
    if n == 0:
        return (0.0, 0.0)
    phat = successes / n
    denom = 1 + z**2 / n
    center = phat + z**2 / (2 * n)
    margin = z * sqrt((phat * (1 - phat) + z**2 / (4 * n)) / n)
    low = (center - margin) / denom
    high = (center + margin) / denom
    return (max(0.0, low), min(1.0, high))


def orthogonality_table(records: list[StepMembership]) -> list[CellCount]:
    """3-way membership table over {drift-fires, policy-fires,
    outcome-says-violation}, with counts + Wilson 95% CIs per cell,
    extending the 2-way drift-vs-policy orthogonality-findings result."""
    n = len(records)
    counts: dict[tuple[bool, bool, bool], int] = {}
    for record in records:
        key = (record.drift_fires, record.policy_fires, record.outcome_violation)
        counts[key] = counts.get(key, 0) + 1

    table: list[CellCount] = []
    for drift_fires in (False, True):
        for policy_fires in (False, True):
            for outcome_violation in (False, True):
                key = (drift_fires, policy_fires, outcome_violation)
                count = counts.get(key, 0)
                low, high = wilson_interval(count, n)
                table.append(
                    CellCount(
                        cell=MembershipCell(drift_fires, policy_fires, outcome_violation),
                        count=count,
                        proportion=count / n if n else 0.0,
                        ci_low=low,
                        ci_high=high,
                    )
                )
    return table
