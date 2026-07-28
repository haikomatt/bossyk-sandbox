"""Activation capture data model + dataset assembly (pure stdlib).

The white-box counterpart to `logprob_metrics`. On the pod, an nnsight job runs
the agent model over each captured decision step and caches residual-stream
activations at chosen layers and timepoints; each becomes an `ActivationRecord`
carrying the same policy label the Gate assigned. Off the pod (here, hermetic),
`assemble_xy` slices those records into the (X, y) matrix the linear probe
consumes.

The timepoints are the drift-probe experiment's T1-T4 matrix (companion doc
`drift-probe-experiment.md`) -- the load-bearing one is **T2** (before the intent
tokens are emitted): a probe that fires there shows the violation is encoded
*before* it is stated, which is the whole point of going to activations rather
than staying at the logprob/text level.

No torch/numpy here: this is just the labelled-record schema and the slicing, so
it stays in the default (dependency-light) test suite. The heavy nnsight/torch
capture is a pod-only script; sklearn probing lives in `probe.py` behind the
`interp` extra.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Timepoint(StrEnum):
    """Where in a step's token stream the residual is cached (drift-probe T1-T4)."""

    T1 = "T1"  # during/after the intent-declaration tokens -- weakest claim
    T2 = "T2"  # BEFORE the intent tokens are emitted -- pre-statement detectability
    T3 = "T3"  # end of the previous step -- earliest useful triage point
    T4 = "T4"  # during action emission, pre-tool-call -- latest intervention point


@dataclass(frozen=True)
class ActivationRecord:
    """One cached residual-stream vector, tagged for supervised probing.

    `activation` is the residual at (`layer`, `timepoint`) for step `step_id`;
    `is_violation` is that step's binary policy label (the Gate/PolicyAwareJudge
    verdict), the probe's target. Kept as a plain float tuple so the schema has
    no array dependency -- the probe converts to a matrix.
    """

    step_id: str
    layer: int
    timepoint: Timepoint
    activation: tuple[float, ...]
    is_violation: bool


def capture_layers(n_layers: int, *, every: int = 2) -> list[int]:
    """Which residual-stream layers to cache: every `every`-th layer (the
    drift-probe sweep is "every 2nd layer"), always including layer 0. Caps cache
    size while still sweeping depth. Raises on non-positive inputs."""
    if n_layers <= 0:
        raise ValueError("n_layers must be positive")
    if every <= 0:
        raise ValueError("every must be positive")
    return list(range(0, n_layers, every))


def assemble_xy(
    records: list[ActivationRecord],
    *,
    layer: int,
    timepoint: Timepoint,
) -> tuple[list[list[float]], list[bool]]:
    """Slice `records` to one (layer, timepoint) cell and return (X, y): X the
    activation rows in record order, y the violation labels. A probe is trained
    per cell, so this is the per-cell view. Raises if the selected activations
    are not all the same dimension (a corrupt capture). Empty selection -> ([], [])."""
    selected = [r for r in records if r.layer == layer and r.timepoint == timepoint]
    x = [list(r.activation) for r in selected]
    y = [r.is_violation for r in selected]
    if x and len({len(row) for row in x}) != 1:
        raise ValueError(
            f"inconsistent activation dims at layer={layer} timepoint={timepoint.value}: "
            f"{sorted({len(row) for row in x})}"
        )
    return x, y
