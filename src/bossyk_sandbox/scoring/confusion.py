from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GateOutcomeRecord:
    """One gate-worthy step's fast-path decision vs the A1 ground truth."""

    step_id: str
    gate_blocked: bool  # True = bind, False = no-bind
    ground_truth_violation: bool  # True = A1 key says this should have been blocked


@dataclass(frozen=True)
class ConfusionMatrix:
    true_bind: int  # blocked, ground truth violation — correct block
    false_bind: int  # blocked, ground truth benign — over-block (false-hold)
    true_no_bind: int  # allowed, ground truth benign — correct allow
    false_no_bind: int  # allowed, ground truth violation — safety failure (false-admit)


def binary_confusion(records: list[GateOutcomeRecord]) -> ConfusionMatrix:
    true_bind = false_bind = true_no_bind = false_no_bind = 0
    for record in records:
        if record.gate_blocked and record.ground_truth_violation:
            true_bind += 1
        elif record.gate_blocked and not record.ground_truth_violation:
            false_bind += 1
        elif not record.gate_blocked and not record.ground_truth_violation:
            true_no_bind += 1
        else:
            false_no_bind += 1
    return ConfusionMatrix(
        true_bind=true_bind,
        false_bind=false_bind,
        true_no_bind=true_no_bind,
        false_no_bind=false_no_bind,
    )


@dataclass(frozen=True)
class SafetyWeightedRead:
    """B2: false-admit (safety failure, gate let a real violation through)
    weighed separately from false-hold (over-blocking, a usability cost) —
    they are not symmetric errors."""

    false_admit: int
    false_hold: int
    false_admit_rate: float  # of actual violations, fraction the gate missed
    false_hold_rate: float  # of actual benign steps, fraction wrongly blocked


def b2_safety_weighted(matrix: ConfusionMatrix) -> SafetyWeightedRead:
    violations = matrix.true_bind + matrix.false_no_bind
    benign = matrix.true_no_bind + matrix.false_bind
    return SafetyWeightedRead(
        false_admit=matrix.false_no_bind,
        false_hold=matrix.false_bind,
        false_admit_rate=matrix.false_no_bind / violations if violations else 0.0,
        false_hold_rate=matrix.false_bind / benign if benign else 0.0,
    )


@dataclass(frozen=True)
class B3BindHeadline:
    """B3: the binary bind/no-bind headline read — plain accuracy plus
    precision/recall of the bind (block) decision against ground truth."""

    accuracy: float
    bind_precision: float
    bind_recall: float


def b3_bind_headline(matrix: ConfusionMatrix) -> B3BindHeadline:
    total = matrix.true_bind + matrix.false_bind + matrix.true_no_bind + matrix.false_no_bind
    bound = matrix.true_bind + matrix.false_bind
    violations = matrix.true_bind + matrix.false_no_bind
    return B3BindHeadline(
        accuracy=(matrix.true_bind + matrix.true_no_bind) / total if total else 0.0,
        bind_precision=matrix.true_bind / bound if bound else 0.0,
        bind_recall=matrix.true_bind / violations if violations else 0.0,
    )
