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


def test_build_gen_payload_forward_offsets_padding_and_text() -> None:
    m = _import()
    d = 4
    # rollout A: 4 gen tokens -> offsets 0,2 valid, 8 out of range (NaN)
    ra = m.Rollout(
        prompt_id=0,
        is_violation=True,
        token_strings=["Sure", " I'll", " modify", " it"],
        residuals={7: [[float(i)] * d for i in range(4)]},
    )
    # rollout B: same prompt, compliant (text-ask), only 2 tokens
    rb = m.Rollout(
        prompt_id=0,
        is_violation=False,
        token_strings=["Could", " you"],
        residuals={7: [[9.0] * d, [8.0] * d]},
    )
    payload = m.build_gen_payload([ra, rb], layers=[7], offsets=[0, 2, 8])

    assert payload["offsets"].tolist() == [0, 2, 8]
    assert payload["is_violation"].tolist() == [True, False]
    assert payload["X_7"].shape == (2, 3, d)
    # rollout A: offset 0 -> token 0; offset 2 -> token 2; offset 8 -> out of range NaN
    assert payload["X_7"][0, 0, 0] == 0.0 and payload["X_7"][0, 1, 0] == 2.0
    assert np.isnan(payload["X_7"][0, 2]).all()
    # rollout B (len 2): offset 0 valid; offsets 2 and 8 out of range -> NaN
    assert payload["X_7"][1, 0, 0] == 9.0
    assert np.isnan(payload["X_7"][1, 1]).all() and np.isnan(payload["X_7"][1, 2]).all()
    # text-so-far accumulates decoded tokens up to and including the offset
    assert payload["text_so_far"][0, 1] == "Sure I'll modify"
    assert payload["text_so_far"][1, 1] == ""  # out of range -> empty
