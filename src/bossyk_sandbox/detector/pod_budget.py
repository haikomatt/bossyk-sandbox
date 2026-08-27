"""Pod budget arithmetic for the detector-training RunPod run
(phase-detector-training.md build-order step 4, "Cost: <= ~$15 GPU spend,
hard-capped in-script; 0 pods left at end.").

Pure arithmetic, no network/no runpodctl -- this module answers "how much
wall-clock time may the pod run for" and "should it stop now", so the
answers are hermetically testable. The actual pod lifecycle (provision,
poll, terminate) lives in `scripts/detector_pod_runner.py`, which imports
this module for its stop/go decisions and never re-derives the arithmetic
inline.

Two layered limits, per the brief:
- `MAX_WALL_SECONDS` (6h): a generous, deliberately-low wall-clock backstop
  ("floor generously -- set max wall at 6h which is under $3 at A40 rates").
- `HARD_CAP_USD` ($15): the absolute backstop. `wall_budget_seconds` takes
  whichever of the two implies less time, so a rate assumption that's too
  low can never let the run exceed the dollar cap.
"""

from __future__ import annotations

from dataclasses import dataclass

# Documented A40 on-demand rate (matches interp-rigorous-run-runbook.md's
# "~$0.44/hr A40"). Used only to translate the dollar cap into a wall-clock
# budget and to estimate spend as the run progresses -- never billed
# directly; RunPod's own metering is the source of truth for actual cost.
A40_HOURLY_USD = 0.44

HARD_CAP_USD = 15.0

# "floor generously -- set max wall at 6h which is under $3 at A40 rates; the
# $15 cap is the absolute backstop."
MAX_WALL_SECONDS = 6 * 3600

SECONDS_PER_HOUR = 3600.0


def cost_estimate_usd(elapsed_seconds: float, *, hourly_usd: float = A40_HOURLY_USD) -> float:
    """Estimated spend for `elapsed_seconds` of pod wall time at `hourly_usd`."""
    if elapsed_seconds < 0:
        raise ValueError(f"elapsed_seconds must be >= 0, got {elapsed_seconds}")
    return elapsed_seconds / SECONDS_PER_HOUR * hourly_usd


def wall_budget_seconds(
    *,
    hard_cap_usd: float = HARD_CAP_USD,
    hourly_usd: float = A40_HOURLY_USD,
    max_wall_seconds: int = MAX_WALL_SECONDS,
) -> int:
    """The wall-clock budget in seconds: the smaller of the 6h floor-generous
    backstop and however many seconds the dollar cap implies at `hourly_usd`
    -- so the effective limit is always the tighter of the two, and a
    too-low rate assumption can never let the run drift past the dollar
    cap."""
    if hard_cap_usd <= 0:
        raise ValueError(f"hard_cap_usd must be > 0, got {hard_cap_usd}")
    if hourly_usd <= 0:
        raise ValueError(f"hourly_usd must be > 0, got {hourly_usd}")
    cap_implied_seconds = hard_cap_usd / hourly_usd * SECONDS_PER_HOUR
    return int(min(max_wall_seconds, cap_implied_seconds))


@dataclass(frozen=True)
class TerminationDecision:
    """Result of `should_terminate`: whether to stop the pod now, and why --
    the pod runner logs `reason` verbatim rather than re-deriving it, so the
    lifecycle log always states which backstop fired (if any)."""

    terminate: bool
    reason: str
    elapsed_seconds: float
    estimated_cost_usd: float


def should_terminate(
    elapsed_seconds: float,
    *,
    hard_cap_usd: float = HARD_CAP_USD,
    hourly_usd: float = A40_HOURLY_USD,
    max_wall_seconds: int = MAX_WALL_SECONDS,
) -> TerminationDecision:
    """Whether the pod should be terminated right now. Two independent
    tripwires, checked every poll by the pod runner:

    1. wall-clock: `elapsed_seconds >= wall_budget_seconds(...)`
    2. dollar: `cost_estimate_usd(elapsed_seconds) >= hard_cap_usd`

    These agree by construction (`wall_budget_seconds` is derived from the
    same two inputs), but both are checked explicitly rather than relying on
    that derivation holding under future edits -- the dollar check is the
    named "absolute backstop" and must never depend on the wall-clock
    branch being correct."""
    budget = wall_budget_seconds(
        hard_cap_usd=hard_cap_usd, hourly_usd=hourly_usd, max_wall_seconds=max_wall_seconds
    )
    cost = cost_estimate_usd(elapsed_seconds, hourly_usd=hourly_usd)

    if cost >= hard_cap_usd:
        return TerminationDecision(
            terminate=True,
            reason=f"dollar cap reached: estimated ${cost:.2f} >= ${hard_cap_usd:.2f} hard cap",
            elapsed_seconds=elapsed_seconds,
            estimated_cost_usd=cost,
        )
    if elapsed_seconds >= budget:
        return TerminationDecision(
            terminate=True,
            reason=(
                f"wall-clock budget reached: {elapsed_seconds:.0f}s >= {budget}s budget "
                f"(estimated ${cost:.2f})"
            ),
            elapsed_seconds=elapsed_seconds,
            estimated_cost_usd=cost,
        )
    return TerminationDecision(
        terminate=False,
        reason=f"within budget: {elapsed_seconds:.0f}s/{budget}s, ${cost:.2f}/${hard_cap_usd:.2f}",
        elapsed_seconds=elapsed_seconds,
        estimated_cost_usd=cost,
    )


def remaining_seconds(
    elapsed_seconds: float,
    *,
    hard_cap_usd: float = HARD_CAP_USD,
    hourly_usd: float = A40_HOURLY_USD,
    max_wall_seconds: int = MAX_WALL_SECONDS,
) -> float:
    """Seconds left in the budget (never negative) -- used by the pod runner
    to decide whether there's plausibly enough time left to start another
    training cell before polling again."""
    budget = wall_budget_seconds(
        hard_cap_usd=hard_cap_usd, hourly_usd=hourly_usd, max_wall_seconds=max_wall_seconds
    )
    return max(0.0, budget - elapsed_seconds)
