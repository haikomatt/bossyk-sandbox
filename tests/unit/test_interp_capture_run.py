"""Hermetic tests for interp.capture_run -- fake tracer, no torch/nnsight."""

from __future__ import annotations

import math

from bossyk_sandbox.interp.activation_capture import Timepoint
from bossyk_sandbox.interp.capture_run import (
    DecisionItem,
    capture_records,
    run_capture_report,
)


def _fake_tracer(prompt: str, layers: list[int]) -> dict[int, list[float]]:
    """Synthetic residuals: a violation prompt (marked with 'VIOL') sits high on
    dim 0, compliant low, with small per-layer jitter. Deterministic, no model."""
    base = 3.0 if "VIOL" in prompt else -3.0
    return {layer: [base + 0.01 * layer, 0.5, -0.5] for layer in layers}


def _items(n: int = 40) -> list[DecisionItem]:
    return [
        DecisionItem(
            step_id=f"s{i}",
            prompt=("VIOL" if i % 2 == 0 else "ok"),
            is_violation=(i % 2 == 0),
        )
        for i in range(n)
    ]


def test_capture_records_one_per_item_and_layer_in_order() -> None:
    items = _items(4)
    records = capture_records(items, [0, 2], _fake_tracer)
    assert len(records) == 4 * 2
    # records for layer 0 are in item order
    layer0 = [r for r in records if r.layer == 0]
    assert [r.step_id for r in layer0] == ["s0", "s1", "s2", "s3"]
    assert all(r.timepoint is Timepoint.T4 for r in records)
    assert layer0[0].is_violation is True and layer0[1].is_violation is False


def test_report_recovers_policy_signal_above_shuffled_floor() -> None:
    report = run_capture_report(_items(), [0, 2, 4], _fake_tracer)
    assert report["n_items"] == 40
    assert report["n_violation"] == 20
    layers = report["layers"]
    assert isinstance(layers, dict)
    for layer_scores in layers.values():
        assert layer_scores["policy"] > 0.9  # separable synthetic signal, held out
        assert layer_scores["shuffled"] <= 0.75  # floor
        assert "error" not in layer_scores  # no is_error labels supplied


def test_report_includes_general_failure_confound_when_error_labels_present() -> None:
    # Every item carries is_error -> the confound probe is reported.
    items = [
        DecisionItem(
            step_id=f"s{i}",
            prompt=("VIOL" if i % 2 == 0 else "ok"),
            is_violation=(i % 2 == 0),
            is_error=(i % 3 == 0),
        )
        for i in range(40)
    ]
    report = run_capture_report(items, [0], _fake_tracer)
    layer0 = report["layers"][0]
    assert "error" in layer0
    assert not math.isnan(layer0["policy"])
