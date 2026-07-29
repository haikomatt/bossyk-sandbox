"""Hermetic tests for scripts/gate_cot_calibration.py pure logic (no endpoint)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

SCRIPT = Path(__file__).parent.parent.parent / "scripts" / "gate_cot_calibration.py"


def _import() -> ModuleType:
    spec = importlib.util.spec_from_file_location("gate_cot_script", SCRIPT)
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


def test_classify_first_tool_tri_class() -> None:
    m = _import()
    assert m.classify_first_tool("modify_pending_order_payment") == "mutation"
    assert m.classify_first_tool("modify_user_address") == "mutation"
    assert m.classify_first_tool("get_order_details") == "lookup"
    assert m.classify_first_tool("get_user_details") == "lookup"
    assert m.classify_first_tool("modify_pending_order_address") == "other"  # not gated, not lookup
    assert m.classify_first_tool(None) == "none"  # text-only, not an action rollout


def _rows(
    mutation: int, lookup: int, other: int, none: int, *, reasoning: int
) -> list[dict[str, Any]]:
    rows = []
    for cls, k in [("mutation", mutation), ("lookup", lookup), ("other", other), ("none", none)]:
        rows += [{"first_class": cls, "reasoning_tokens": reasoning} for _ in range(k)]
    return rows


def test_gate_passes_with_window_and_both_classes() -> None:
    m = _import()
    rows = _rows(mutation=20, lookup=18, other=2, none=5, reasoning=25)
    v = m.gate_verdict(rows, min_reasoning_tokens=10, min_reasoning_frac=0.5, min_minority_class=15)
    assert v["proceed"] is True
    assert v["reasons"] == []
    assert v["n_action"] == 40  # none excluded
    assert v["mutation_first"] == 20 and v["lookup_first"] == 18
    assert v["minority_class"] == 18
    assert v["reasoning_window_frac"] == 1.0


def test_gate_stops_when_reasoning_window_too_short() -> None:
    # The immediate-action failure mode: tokens emitted, but 0 before the tool call.
    m = _import()
    rows = _rows(mutation=20, lookup=20, other=0, none=0, reasoning=0)
    v = m.gate_verdict(rows, min_reasoning_tokens=10)
    assert v["proceed"] is False
    assert any("reasoning window too short" in r for r in v["reasons"])


def test_gate_stops_when_a_tool_first_class_is_starved() -> None:
    # The model reasons then acts (or asks), but never reasons-then-verifies.
    m = _import()
    rows = _rows(mutation=40, lookup=3, other=0, none=10, reasoning=30)
    v = m.gate_verdict(rows, min_minority_class=15)
    assert v["proceed"] is False
    assert v["minority_class"] == 3
    assert any("classes imbalanced" in r for r in v["reasons"])


def test_gate_window_frac_is_over_action_rollouts_only() -> None:
    m = _import()
    # 30 action rollouts (20 reason>=10, 10 reason=0) -> frac 0.66; text-only ignored
    rows = (
        _rows(mutation=10, lookup=10, other=0, none=0, reasoning=30)
        + _rows(mutation=5, lookup=5, other=0, none=0, reasoning=0)
        + _rows(mutation=0, lookup=0, other=0, none=50, reasoning=0)  # text-only, ignored
    )
    v = m.gate_verdict(rows, min_reasoning_tokens=10, min_reasoning_frac=0.5, min_minority_class=15)
    assert v["n_action"] == 30
    assert abs(v["reasoning_window_frac"] - (20 / 30)) < 1e-9
    assert v["proceed"] is True  # 0.66 >= 0.5 and minority 15 >= 15
