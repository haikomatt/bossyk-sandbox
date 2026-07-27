"""Tool-call smoke harness: the pure classifier
(`parse_tool_call_outcome`) that turns "did the model tool-call, and did
the configured parser structure it?" into a first-class finding instead of
a crash or a silent mis-score. `scripts/smoke_tool_call.py`'s `main()` makes
a REAL, billable network call when actually run (gated on RUN_TOOL_SMOKE=1)
-- this test suite only imports the module (side-effect-free at import
time, mirrors test_live_h2h4_bench_script.py's pattern) and exercises the
classifier against fixtures; it never calls `main()`.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

SCRIPT_PATH = Path(__file__).parent.parent.parent / "scripts" / "smoke_tool_call.py"


def _import_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("smoke_tool_call", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_smoke_tool_call_script_imports_without_network_and_defines_main() -> None:
    module = _import_script()

    assert callable(module.main)


def test_parse_tool_call_outcome_recognizes_a_parsed_hermes_style_tool_call() -> None:
    # Qwen/hermes: the API returns a properly structured tool_calls entry.
    module = _import_script()
    response = {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "cancel_pending_order",
                                "arguments": '{"order_id": "#W1"}',
                            },
                        }
                    ],
                }
            }
        ]
    }

    outcome = module.parse_tool_call_outcome(response)

    assert outcome == {"parsed": True, "raw_text_tool_call": False}


def test_parse_tool_call_outcome_recognizes_an_unparsed_python_tag_text_response() -> None:
    # Llama-3.2-style raw-text tool call: the model attempted one, but the
    # configured parser never structured it into message.tool_calls -- a
    # first-class "can't-tool-call" finding, not a crash.
    module = _import_script()
    response = {
        "choices": [
            {
                "message": {
                    "content": '<|python_tag|>cancel_pending_order.call(order_id="#W1")',
                    "tool_calls": None,
                }
            }
        ]
    }

    outcome = module.parse_tool_call_outcome(response)

    assert outcome == {"parsed": False, "raw_text_tool_call": True}


def test_parse_tool_call_outcome_is_neither_when_the_model_just_declines_in_plain_text() -> None:
    # A model that refuses in plain prose engaged with neither the tool API
    # nor a raw-text tool-call sentinel -- distinct from both other outcomes.
    module = _import_script()
    response = {
        "choices": [
            {
                "message": {
                    "content": "I can't do that without verifying your identity.",
                    "tool_calls": None,
                }
            }
        ]
    }

    outcome = module.parse_tool_call_outcome(response)

    assert outcome == {"parsed": False, "raw_text_tool_call": False}


def test_parse_tool_call_outcome_treats_an_empty_tool_calls_list_as_unparsed() -> None:
    # Some providers return "tool_calls": [] rather than omitting/nulling
    # the field -- must not be mistaken for a parsed call.
    module = _import_script()
    response = {"choices": [{"message": {"content": "no tools needed", "tool_calls": []}}]}

    outcome = module.parse_tool_call_outcome(response)

    assert outcome == {"parsed": False, "raw_text_tool_call": False}
