"""Activation-capture orchestration (pure given an injected tracer).

The library half of the on-pod capture step: turn a list of labelled decision
prompts into `ActivationRecord`s and then a per-layer probe report, without
importing torch/nnsight. The one heavy dependency -- running the model to read
its residual stream -- is an INJECTED `tracer` callable, exactly the seam
`runpod_serving`/`interp_pod` use for `run`: production passes an nnsight-backed
tracer (`scripts/interp_capture.py`); tests pass a fake returning synthetic
vectors. So this orchestration stays in the default (torch-free) test suite.

MVP timepoint is T4 (the last context token, pre-action) -- a single, unambiguous
position that needs no tokenizer-specific span-finding. The full T1-T3 sweep
(pre-statement detectability et al.) is the documented refinement once real
tokens are in hand (drift-probe open item #2).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypedDict

from bossyk_sandbox.interp.activation_capture import (
    ActivationRecord,
    Timepoint,
    assemble_xy,
)
from bossyk_sandbox.interp.probe import probe_auroc, shuffled_label_auroc


class CaptureReport(TypedDict):
    """Per-layer probe report for one capture run."""

    timepoint: str
    n_items: int
    n_violation: int
    layers: dict[int, dict[str, float]]  # layer -> {policy, shuffled, error?}


# A tracer reads the model's residual stream: (prompt, layers) -> {layer:
# residual vector at the target token}. nnsight-backed in production, fake in
# tests. The ONLY thing that touches torch/the GPU.
Tracer = Callable[[str, list[int]], dict[int, list[float]]]


@dataclass(frozen=True)
class DecisionItem:
    """One labelled decision to capture: the agent's prompt context at a scored
    step, its binary policy label, and optionally an `is_error` label (a general
    incoherence/failure flag) so the report can run the general-failure confound
    control alongside the policy probe.

    `action` is the agent's OWN response at that turn (text + tool calls). It is
    carried through the dataset so the coherence judge -- which sets `is_error`
    from a quality judgment on the action -- can run as a separate, re-runnable,
    auditable step off the agent-driving loop. The capture/probe never reads it;
    it is provenance for the confound label."""

    step_id: str
    prompt: str
    is_violation: bool
    is_error: bool | None = None
    action: str | None = None


def capture_records(
    items: list[DecisionItem],
    layers: list[int],
    tracer: Tracer,
    *,
    timepoint: Timepoint = Timepoint.T4,
) -> list[ActivationRecord]:
    """Run `tracer` over each item and emit one ActivationRecord per (item,
    layer). Records are in item order for each layer, so a label vector built as
    `[item.<label> for item in items]` lines up with `assemble_xy`'s output."""
    records: list[ActivationRecord] = []
    for item in items:
        activations = tracer(item.prompt, layers)
        for layer in layers:
            records.append(
                ActivationRecord(
                    step_id=item.step_id,
                    layer=layer,
                    timepoint=timepoint,
                    activation=tuple(activations[layer]),
                    is_violation=item.is_violation,
                )
            )
    return records


def report_from_records(
    records: list[ActivationRecord],
    items: list[DecisionItem],
    layers: list[int],
    *,
    timepoint: Timepoint = Timepoint.T4,
    seed: int = 0,
) -> CaptureReport:
    """Probe each layer from ALREADY-CAPTURED records (no tracer, no GPU). Split
    out from `run_capture_report` so the on-pod script captures ONCE, persists the
    activations, then probes -- the authoritative, split-robust re-probe runs
    off-pod from the saved npz (scripts/reprobe.py). This single-split report stays
    a quick on-pod sanity check only."""
    have_error = bool(items) and all(item.is_error is not None for item in items)

    per_layer: dict[int, dict[str, float]] = {}
    for layer in layers:
        x, y = assemble_xy(records, layer=layer, timepoint=timepoint)
        scores = {
            "policy": probe_auroc(x, y, seed=seed),
            "shuffled": shuffled_label_auroc(x, y, seed=seed),
        }
        if have_error:
            error_labels = [bool(item.is_error) for item in items]
            scores["error"] = probe_auroc(x, error_labels, seed=seed)
        per_layer[layer] = scores

    return {
        "timepoint": timepoint.value,
        "n_items": len(items),
        "n_violation": sum(1 for item in items if item.is_violation),
        "layers": per_layer,
    }


def run_capture_report(
    items: list[DecisionItem],
    layers: list[int],
    tracer: Tracer,
    *,
    timepoint: Timepoint = Timepoint.T4,
    seed: int = 0,
) -> CaptureReport:
    """Capture activations and probe each layer. Per layer, reports held-out
    probe AUROC for the policy label, the shuffled-label floor, and -- when every
    item carries an `is_error` label -- the general-failure confound probe. The
    policy result is only meaningful where it beats both the shuffled floor AND
    the error probe."""
    records = capture_records(items, layers, tracer, timepoint=timepoint)
    return report_from_records(records, items, layers, timepoint=timepoint, seed=seed)
