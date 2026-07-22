from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

Z_95 = 1.959963984540054


@dataclass(frozen=True)
class StepMembership:
    """One scored step's membership in the three orthogonality axes.

    `drift_fires` / `policy_fires` are tri-state (Finding 4): `None` means
    the judge verdict for that instrument is unavailable for this step
    (error, unscored, or the instrument never ran) rather than a confirmed
    non-fire. `outcome_violation` is authored ground truth and is always
    known, so it stays plain `bool`. Callers must resolve the `None` case
    (see `split_complete`) before feeding records to `orthogonality_table`.
    """

    step_id: str
    drift_fires: bool | None
    policy_fires: bool | None
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


def split_complete(
    records: list[StepMembership],
) -> tuple[list[StepMembership], list[StepMembership]]:
    """Partitions `records` into (complete, incomplete): complete records
    have a real bool for both `drift_fires` and `policy_fires` (Finding 4).
    Order-preserving. Callers must split before calling `orthogonality_table`,
    which refuses incomplete records rather than silently miscounting a
    judge outage as a non-fire."""
    raise NotImplementedError


def orthogonality_table(records: list[StepMembership]) -> list[CellCount]:
    """3-way membership table over {drift-fires, policy-fires,
    outcome-says-violation}, with counts + Wilson 95% CIs per cell,
    extending the 2-way drift-vs-policy orthogonality-findings result.

    Requires every record to be complete (see `split_complete`) -- raises
    `ValueError` naming the offending step_id(s) rather than silently
    dropping a `None` field into an undercounted cell."""
    n = len(records)
    counts: dict[tuple[bool | None, bool | None, bool], int] = {}
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
