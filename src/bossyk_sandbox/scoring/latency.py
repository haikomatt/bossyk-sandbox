from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

# The wall-clock seam (plan §15B): every caller that wants latency numbers
# passes `clock`, defaulting to the real `time.perf_counter`. Deterministic
# tests inject a fake clock (e.g. an iterator's `__next__`, or a small
# counter closure) so latency assertions never depend on real elapsed time
# or introduce flakiness under load.
Clock = Callable[[], float]


@dataclass(frozen=True)
class LatencyRecord:
    """One timed call's wall-clock cost, in seconds."""

    instrument: str
    elapsed_s: float


def timed[T](
    instrument: str, fn: Callable[[], T], *, clock: Clock = time.perf_counter
) -> tuple[T, LatencyRecord]:
    """Runs `fn`, timing it with `clock`. `elapsed_s` is floored at 0.0 so a
    pathological fake clock (end < start) can't produce a negative latency
    that would corrupt an aggregate."""
    start = clock()
    result = fn()
    end = clock()
    return result, LatencyRecord(instrument=instrument, elapsed_s=max(0.0, end - start))


@dataclass(frozen=True)
class LatencySummary:
    """Per-instrument wall-clock summary: count plus mean/p50/p95/max,
    in seconds -- the §15B "latency percentiles" exit-criteria shape."""

    instrument: str
    count: int
    mean_s: float
    p50_s: float
    p95_s: float
    max_s: float


def _percentile(ordered: list[float], fraction: float) -> float:
    """Nearest-rank percentile over an already-sorted list. `ordered` must
    be non-empty; callers (`summarize_latency`) only invoke this per
    instrument group, which is never empty by construction."""
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[index]


def summarize_latency(records: list[LatencyRecord]) -> dict[str, LatencySummary]:
    """Groups `records` by instrument and computes count/mean/p50/p95/max
    wall-clock seconds. Pure math over already-collected `LatencyRecord`s --
    no timing happens here, so this is trivially deterministic to test."""
    by_instrument: dict[str, list[float]] = {}
    for record in records:
        by_instrument.setdefault(record.instrument, []).append(record.elapsed_s)

    summaries: dict[str, LatencySummary] = {}
    for instrument, values in by_instrument.items():
        ordered = sorted(values)
        summaries[instrument] = LatencySummary(
            instrument=instrument,
            count=len(ordered),
            mean_s=sum(ordered) / len(ordered),
            p50_s=_percentile(ordered, 0.50),
            p95_s=_percentile(ordered, 0.95),
            max_s=ordered[-1],
        )
    return summaries
