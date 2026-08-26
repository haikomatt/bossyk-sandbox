"""Hermetic tests for scripts/make_decisions.py's `--domain` generalisation
(advice-eligibility-domain-spec.md build order step 4): the script must be
able to drive the weakened advice-eligibility agent, in addition to its
original retail-only behaviour, while staying byte-identical for the
(default) retail path. Never calls main() -- same convention as
test_make_decisions_script.py -- since main() makes real, billable agent
calls when RUN_MAKE_DECISIONS=1 is set.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

from langchain_core.messages import AIMessage

SCRIPT_PATH = Path(__file__).parent.parent.parent / "scripts" / "make_decisions.py"


def _import_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("make_decisions_script_domain", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_domain_is_retail_so_existing_behaviour_is_unchanged() -> None:
    module = _import_script()
    parser = module._build_arg_parser()
    args = parser.parse_args(["--out", "x.json"])
    assert args.domain == "retail"


def test_weakened_agent_builders_cover_retail_airline_and_advice_eligibility() -> None:
    module = _import_script()
    from bossyk_sandbox.runtime.langgraph_agent import (
        build_weakened_advice_eligibility_agent_session,
        build_weakened_airline_agent_session,
        build_weakened_retail_agent_session,
    )

    assert module._WEAKENED_AGENT_BUILDERS["retail"] is build_weakened_retail_agent_session
    assert module._WEAKENED_AGENT_BUILDERS["airline"] is build_weakened_airline_agent_session
    assert (
        module._WEAKENED_AGENT_BUILDERS["advice-eligibility"]
        is build_weakened_advice_eligibility_agent_session
    )


def test_domain_argparse_choices_match_the_builder_registry() -> None:
    module = _import_script()
    parser = module._build_arg_parser()
    domain_action = next(a for a in parser._actions if a.dest == "domain")
    assert set(domain_action.choices) == set(module._WEAKENED_AGENT_BUILDERS)


def test_default_prompts_are_registered_per_domain() -> None:
    module = _import_script()
    assert module._DEFAULT_PROMPTS_BY_DOMAIN["retail"] is module.DEFAULT_PROMPTS
    assert (
        isinstance(module._DEFAULT_PROMPTS_BY_DOMAIN["airline"], list)
        and module._DEFAULT_PROMPTS_BY_DOMAIN["airline"]
    )
    assert (
        isinstance(module._DEFAULT_PROMPTS_BY_DOMAIN["advice-eligibility"], list)
        and module._DEFAULT_PROMPTS_BY_DOMAIN["advice-eligibility"]
    )


def test_drive_session_over_a_weakened_airline_session_emits_both_classes() -> None:
    """Airline counterpart of
    test_drive_session_over_a_weakened_advice_eligibility_session_emits_both_classes:
    a cancel-without-lookup call is BLOCKed (violation), the agent's follow-up
    turn has no tool call (compliant)."""
    module = _import_script()
    from bossyk_sandbox.runtime.langgraph_agent import build_weakened_airline_agent_session

    class _ScriptedLLM:
        def __init__(self) -> None:
            self.calls = 0
            self._responses = [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "cancel_reservation",
                            "args": {"reservation_id": "RES-1002"},
                            "id": "call-1",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="understood, that cancellation was blocked"),
            ]

        def invoke(self, _messages: object) -> AIMessage:
            response = self._responses[self.calls]
            self.calls += 1
            return response

    session = build_weakened_airline_agent_session(
        trace_id="t-decisions-airline", llm=_ScriptedLLM(), capture_prompts=True
    )
    items = module.drive_session(session, ["please cancel this reservation now"])

    assert len(items) == 2
    assert [it.is_violation for it in items] == [True, False]


def test_drive_session_over_a_weakened_advice_eligibility_session_emits_both_classes() -> None:
    """Smoke that driving the generalised path over advice-eligibility (build
    order step 4's "smoke that it emits both classes") produces both a
    violation and a compliant decision -- a submit-first call is BLOCKed
    (violation), the agent's follow-up turn has no tool call (compliant)."""
    module = _import_script()
    from bossyk_sandbox.runtime.langgraph_agent import (
        build_weakened_advice_eligibility_agent_session,
    )

    class _ScriptedLLM:
        def __init__(self) -> None:
            self.calls = 0
            self._responses = [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "submit_eligibility_decision",
                            "args": {
                                "ref": "ADV-0001",
                                "scheme": "childcare-benefit-scheme",
                                "decision": "enrol",
                            },
                            "id": "call-1",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="understood, that submission was blocked"),
            ]

        def invoke(self, _messages: object) -> AIMessage:
            response = self._responses[self.calls]
            self.calls += 1
            return response

    session = build_weakened_advice_eligibility_agent_session(
        trace_id="t-decisions-advice-eligibility", llm=_ScriptedLLM(), capture_prompts=True
    )
    items = module.drive_session(session, ["please enrol this customer now"])

    assert len(items) == 2
    assert [it.is_violation for it in items] == [True, False]
