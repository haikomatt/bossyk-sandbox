"""Run-2 sizing calculator (hermetic diversity fix, item 4:
phase-detector-training-step2-datagen.md Issues & Fixes / Part B run 1 fix
plan, step 3). Pure arithmetic -- no network, no LLM, no filesystem.

Feeds the run-2 spend-authorization ask: given a small calibration sample's
MEASURED per-domain unique-rate (unique decisions / calls, after dedupe) and
violation base rate (measured over UNIQUE decisions -- see run-1's finding
that unique-level violation rates were ~4% retail / ~8% airline / ~20%
advice-eligibility, well off the ~17-18% the plan originally assumed) plus
the REAL per-call cost (from the provider dashboard, not the placeholder
estimate), computes the call volume and $ needed to reach a target positives-
per-cell -- the dossier's pre-registered ~150-300 power floor
(a-fine-tuned-violation-detector-transfers-cross-domain.md "Sample size /
power").
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SizeEstimate:
    domain: str
    target_positives: int
    unique_rate: float
    violation_rate: float
    cost_per_call_usd: float
    unique_decisions_needed: int
    calls_needed: int
    estimated_cost_usd: float


def size_run(
    *,
    domain: str,
    unique_rate: float,
    violation_rate: float,
    cost_per_call_usd: float,
    target_positives: int,
) -> SizeEstimate:
    """How many calls (and how much $) are needed to reach `target_positives`
    violations for `domain`, given a MEASURED `unique_rate` (unique decisions
    / calls) and `violation_rate` (violations / unique decisions, i.e.
    measured over UNIQUE decisions -- matching how run-1's per-domain
    base-rate finding was measured). Both rates and the cost must come from
    real measurement (a calibration sample, the provider's billing
    dashboard), never guessed -- this function does no measuring of its own,
    only the arithmetic.

    `unique_decisions_needed = ceil(target_positives / violation_rate)`;
    `calls_needed = ceil(unique_decisions_needed / unique_rate)`; cost is
    `calls_needed * cost_per_call_usd`. Both divisions round UP (ceiling, not
    floor/round) so the estimate is always sufficient, never short."""
    if not (0.0 < unique_rate <= 1.0):
        raise ValueError(f"unique_rate must be in (0, 1], got {unique_rate}")
    if not (0.0 < violation_rate <= 1.0):
        raise ValueError(f"violation_rate must be in (0, 1], got {violation_rate}")
    if cost_per_call_usd < 0.0:
        raise ValueError(f"cost_per_call_usd must be >= 0, got {cost_per_call_usd}")
    if target_positives <= 0:
        raise ValueError(f"target_positives must be > 0, got {target_positives}")

    unique_decisions_needed = math.ceil(target_positives / violation_rate)
    calls_needed = math.ceil(unique_decisions_needed / unique_rate)
    estimated_cost_usd = calls_needed * cost_per_call_usd

    return SizeEstimate(
        domain=domain,
        target_positives=target_positives,
        unique_rate=unique_rate,
        violation_rate=violation_rate,
        cost_per_call_usd=cost_per_call_usd,
        unique_decisions_needed=unique_decisions_needed,
        calls_needed=calls_needed,
        estimated_cost_usd=estimated_cost_usd,
    )
