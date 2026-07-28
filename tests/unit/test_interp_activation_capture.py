"""Hermetic tests for interp.activation_capture -- schema + slicing, no torch."""

from __future__ import annotations

import pytest

from bossyk_sandbox.interp.activation_capture import (
    ActivationRecord,
    Timepoint,
    assemble_xy,
    capture_layers,
)


def _rec(
    step_id: str, layer: int, tp: Timepoint, act: tuple[float, ...], viol: bool
) -> ActivationRecord:
    return ActivationRecord(
        step_id=step_id, layer=layer, timepoint=tp, activation=act, is_violation=viol
    )


def test_timepoint_values() -> None:
    assert [t.value for t in Timepoint] == ["T1", "T2", "T3", "T4"]


def test_capture_layers_every_second_including_zero() -> None:
    assert capture_layers(8) == [0, 2, 4, 6]
    assert capture_layers(5, every=1) == [0, 1, 2, 3, 4]


def test_capture_layers_rejects_nonpositive() -> None:
    with pytest.raises(ValueError):
        capture_layers(0)
    with pytest.raises(ValueError):
        capture_layers(4, every=0)


def test_assemble_xy_filters_to_layer_and_timepoint_in_order() -> None:
    records = [
        _rec("s1", 4, Timepoint.T2, (0.1, 0.2), True),
        _rec("s1", 4, Timepoint.T1, (9.0, 9.0), False),  # wrong timepoint
        _rec("s2", 2, Timepoint.T2, (5.0, 5.0), False),  # wrong layer
        _rec("s3", 4, Timepoint.T2, (0.3, 0.4), False),
    ]
    x, y = assemble_xy(records, layer=4, timepoint=Timepoint.T2)
    assert x == [[0.1, 0.2], [0.3, 0.4]]
    assert y == [True, False]


def test_assemble_xy_empty_selection() -> None:
    records = [_rec("s1", 0, Timepoint.T1, (1.0,), True)]
    assert assemble_xy(records, layer=99, timepoint=Timepoint.T4) == ([], [])


def test_assemble_xy_rejects_inconsistent_dims() -> None:
    records = [
        _rec("s1", 4, Timepoint.T2, (0.1, 0.2), True),
        _rec("s2", 4, Timepoint.T2, (0.1, 0.2, 0.3), False),  # different dim
    ]
    with pytest.raises(ValueError, match="inconsistent activation dims"):
        assemble_xy(records, layer=4, timepoint=Timepoint.T2)
