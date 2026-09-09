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


def test_build_gen_payload_tool_call_anchor_is_backward_and_action_only() -> None:
    # The CoT lead-time regime: a reasoning window precedes the tool call, so we
    # anchor BACKWARD from the tool-call token over ACTION rollouts only (both
    # mutation-first and lookup-first have a tool-call anchor; a text-only
    # compliant rollout has none and is excluded).
    m = _import()
    d = 2
    # rollout A: mutation-first, tool call at generated token index 3
    ra = m.Rollout(
        prompt_id=0,
        is_violation=True,
        token_strings=["I'll", " just", " cancel", " <tool_call>", "{"],
        residuals={7: [[float(i)] * d for i in range(5)]},
    )
    # rollout B: lookup-first (compliant), same prompt, tool call at index 3
    rb = m.Rollout(
        prompt_id=0,
        is_violation=False,
        token_strings=["Let", " me", " check", " <tool_call>", "{"],
        residuals={7: [[10.0 + i] * d for i in range(5)]},
    )
    # rollout C: text-only compliant, NO tool call -> excluded under tool_call anchor
    rc = m.Rollout(
        prompt_id=0,
        is_violation=False,
        token_strings=["Could", " you", " confirm"],
        residuals={7: [[99.0] * d for _ in range(3)]},
    )
    payload = m.build_gen_payload(
        [ra, rb, rc], layers=[7], offsets=[-1, -2, -4], anchor="tool_call"
    )

    # offsets stored ascending; only the two action rollouts survive
    assert payload["offsets"].tolist() == [-4, -2, -1]
    assert payload["is_violation"].tolist() == [True, False]
    assert payload["prompt_id"].tolist() == [0, 0]
    assert payload["X_7"].shape == (2, 3, d)
    # rollout A, anchor index 3: col -4 -> pos -1 (NaN); col -2 -> pos 1; col -1 -> pos 2
    assert np.isnan(payload["X_7"][0, 0]).all()
    assert payload["X_7"][0, 1, 0] == 1.0
    assert payload["X_7"][0, 2, 0] == 2.0
    # rollout B, anchor index 3: col -2 -> pos 1 (11.0); col -1 -> pos 2 (12.0)
    assert payload["X_7"][1, 1, 0] == 11.0
    assert payload["X_7"][1, 2, 0] == 12.0
    # text-so-far is the reasoning up to and including each pre-tool-call position
    assert payload["text_so_far"][0, 2] == "I'll just cancel"
    assert payload["text_so_far"][0, 1] == "I'll just"
    assert payload["text_so_far"][0, 0] == ""  # position before generation start


def test_build_gen_payload_emits_called_tool_label() -> None:
    """The payload carries a `called_tool` label alongside `is_violation`.

    Needed to probe the label used by the pre-generation tool-call decoding
    literature (arXiv 2605.09252, 2604.01202), which asks "will a tool be
    called", not "will this be a policy violation". Under the `start` anchor
    both classes are present, so the two labels are genuinely different: a
    lookup-first rollout calls a tool without violating.
    """
    m = _import()
    d = 2
    # lookup-first: calls a tool, does NOT violate
    lookup = m.Rollout(
        prompt_id=0,
        is_violation=False,
        token_strings=["Let", " me", " <tool_call>", "{"],
        residuals={7: [[float(i)] * d for i in range(4)]},
    )
    # text-only compliant: no tool call at all
    text_only = m.Rollout(
        prompt_id=0,
        is_violation=False,
        token_strings=["Could", " you", " confirm"],
        residuals={7: [[9.0] * d for _ in range(3)]},
    )
    # mutation-first: calls a tool AND violates
    mutation = m.Rollout(
        prompt_id=1,
        is_violation=True,
        token_strings=["Sure", " <tool_call>", "{"],
        residuals={7: [[1.0] * d for _ in range(3)]},
    )

    payload = m.build_gen_payload([lookup, text_only, mutation], layers=[7], offsets=[0])

    assert payload["called_tool"].tolist() == [True, False, True]
    assert payload["is_violation"].tolist() == [False, False, True]
    # the two labels genuinely differ: rollout 0 calls a tool without violating
    assert payload["called_tool"][0] != payload["is_violation"][0]
