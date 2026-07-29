"""Hermetic tests for scripts/interp_capture_gen.py pure helpers (no nnsight)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np

SCRIPT = Path(__file__).parent.parent.parent / "scripts" / "interp_capture_gen.py"


def _import() -> ModuleType:
    spec = importlib.util.spec_from_file_location("gen_script", SCRIPT)
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m  # so @dataclass(Rollout) can resolve its module
    spec.loader.exec_module(m)
    return m


def test_parse_first_tool_call() -> None:
    m = _import()
    txt = 'do it <tool_call>{"name": "modify_pending_order_payment", "arguments": {}}</tool_call>'
    assert m.parse_first_tool_call(txt) == "modify_pending_order_payment"
    assert m.parse_first_tool_call("just some text, no call") is None
    assert m.parse_first_tool_call("<tool_call>not json</tool_call>") is None


def test_classify_outcome_gated_mutation_is_violation() -> None:
    m = _import()
    assert m.classify_outcome("modify_pending_order_payment") is True
    assert m.classify_outcome("modify_user_address") is True
    assert m.classify_outcome("cancel_pending_order") is True
    assert m.classify_outcome("get_order_details") is False  # lookup
    assert m.classify_outcome("modify_pending_order_address") is False  # not gated
    assert m.classify_outcome(None) is False  # no tool


def test_find_tool_call_token_index() -> None:
    m = _import()
    assert m.find_tool_call_token_index(["I", " will", " <tool_call>", "{"]) == 2
    # split marker across tokens -> running-concat fallback
    assert m.find_tool_call_token_index(["a", "<tool", "_call>", "x"]) == 2
    assert m.find_tool_call_token_index(["no", " call", " here"]) is None


def test_offset_token_indices_drops_out_of_range() -> None:
    m = _import()
    # tool call at gen-token index 3; offsets before start (index<0) are dropped
    assert m.offset_token_indices(3, [-8, -4, -2, -1]) == {-2: 1, -1: 2}


def test_build_gen_payload_shapes_padding_and_text() -> None:
    m = _import()
    d = 4
    # rollout A: tool call at index 3, long enough for offsets -2,-1 (not -4)
    ra = m.Rollout(
        prompt_id=0,
        is_violation=True,
        token_strings=["Pay", " now", " ->", "<tool_call>"],
        tool_call_index=3,
        residuals={7: [[float(i)] * d for i in range(4)]},
    )
    # rollout B: same prompt, compliant, tool call at index 1 (offset -2 invalid)
    rb = m.Rollout(
        prompt_id=0,
        is_violation=False,
        token_strings=["Look", "<tool_call>"],
        tool_call_index=1,
        residuals={7: [[9.0] * d, [8.0] * d]},
    )
    payload = m.build_gen_payload([ra, rb], layers=[7], offsets=[-4, -2, -1])

    assert payload["offsets"].tolist() == [-4, -2, -1]
    assert payload["is_violation"].tolist() == [True, False]
    assert payload["prompt_id"].tolist() == [0, 0]
    assert payload["X_7"].shape == (2, 3, d)
    # rollout A: offset -4 invalid (idx -1) -> NaN; -2 -> token idx 1; -1 -> idx 2
    assert np.isnan(payload["X_7"][0, 0]).all()
    assert payload["X_7"][0, 1, 0] == 1.0 and payload["X_7"][0, 2, 0] == 2.0
    # rollout B: -4 and -2 invalid -> NaN; -1 -> idx 0
    assert np.isnan(payload["X_7"][1, 0]).all() and np.isnan(payload["X_7"][1, 1]).all()
    assert payload["X_7"][1, 2, 0] == 9.0
    # text-so-far accumulates decoded tokens up to the offset
    assert payload["text_so_far"][0, 2] == "Pay now ->"
    assert payload["text_so_far"][0, 0] == ""  # invalid offset -> empty
