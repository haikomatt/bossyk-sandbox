from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command

from bossyk_sandbox.instruments.base import ProposedAction
from bossyk_sandbox.instruments.hardcoded_rule import RequireLookupBeforeCancel
from bossyk_sandbox.runtime.langgraph_agent import (
    AirlineAgentSession,
    build_airline_agent_session,
    build_retail_agent_session,
    build_weakened_retail_agent_session,
    retail_tool_schemas,
    weaken_policy,
)


@dataclass
class _ScriptedLLM:
    """Fake tool-bound chat model: `invoke` pops the next scripted
    `AIMessage` off a queue regardless of what messages it's called with.
    Stands in for a real ChatOpenAI client from the graph's perspective,
    with no network call and no API key."""

    responses: list[AIMessage]
    calls: int = 0

    def invoke(self, _messages: Any) -> AIMessage:
        response = self.responses[self.calls]
        self.calls += 1
        return response


@dataclass
class _CountingToolkit:
    """Fake tau2 toolkit: counts invocations per (tool_name, frozen args)
    and can be configured to raise for specific tool names, to prove a
    failed call never fabricates a successful history entry."""

    raising_tools: frozenset[str] = frozenset()
    counts: dict[tuple[str, tuple[tuple[str, Any], ...]], int] = field(default_factory=dict)

    def get_tools(self) -> dict[str, Any]:
        return {}

    def use_tool(self, name: str, **kwargs: Any) -> str:
        key = (name, tuple(sorted(kwargs.items())))
        self.counts[key] = self.counts.get(key, 0) + 1
        if name in self.raising_tools:
            raise RuntimeError("boom")
        return "ok"

    def call_count(self, name: str, **kwargs: Any) -> int:
        key = (name, tuple(sorted(kwargs.items())))
        return self.counts.get(key, 0)


@dataclass
class _FakeEnvironment:
    """Fake tau2 environment: exposes `.policy` and `.tools`, enough for
    `build_airline_agent_session(environment=...)` to skip the real tau2
    `get_environment()` call entirely."""

    tools: _CountingToolkit
    policy: str = "test policy"


def _tool_call(name: str, args: dict[str, Any], call_id: str) -> dict[str, Any]:
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


def _run_to_completion(
    session: AirlineAgentSession,
    *,
    thread_id: str,
    resume_values: dict[str, str] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Invoke the graph to completion, resuming every interrupt encountered.

    Each interrupt is resumed with `resume_values[tool_call_id]` if present,
    else the interrupt payload's own `auto_verdict` (accepting the automatic
    decision) — mirroring `scripts/live_demo.py`'s driver contract. Returns
    the final graph result plus every interrupt payload observed, in order.
    """
    config = {"configurable": {"thread_id": thread_id}}
    resume_values = resume_values or {}
    payloads: list[dict[str, Any]] = []

    result: dict[str, Any] = session.graph.invoke(  # type: ignore[call-overload]
        {"messages": [HumanMessage(content="please help with my reservation")]}, config=config
    )
    while "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        payloads.append(payload)
        resume = resume_values.get(payload["tool_call_id"], payload["auto_verdict"])
        result = session.graph.invoke(  # type: ignore[call-overload]
            Command(resume=resume), config=config
        )
    return result, payloads


def test_two_tool_calls_in_one_ai_message_each_execute_exactly_once() -> None:
    toolkit = _CountingToolkit()
    llm = _ScriptedLLM(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    _tool_call("get_reservation_details", {"reservation_id": "R1"}, "call-1"),
                    _tool_call("cancel_reservation", {"reservation_id": "R1"}, "call-2"),
                ],
            ),
            AIMessage(content="done"),
        ]
    )
    session = build_airline_agent_session(
        trace_id="t-two-calls", llm=llm, environment=_FakeEnvironment(tools=toolkit)
    )

    _result, payloads = _run_to_completion(session, thread_id="two-calls")

    assert len(payloads) == 2
    assert toolkit.call_count("get_reservation_details", reservation_id="R1") == 1
    assert toolkit.call_count("cancel_reservation", reservation_id="R1") == 1
    assert len(session.steps) == 2
    # The cancel's auto verdict saw the already-executed lookup.
    assert payloads[1]["auto_verdict"] == "allow"
    assert session.gate is not None
    assert session.gate.history == [
        ProposedAction("get_reservation_details", {"reservation_id": "R1"}),
        ProposedAction("cancel_reservation", {"reservation_id": "R1"}),
    ]
    assert all(step.metadata["overridden"] is False for step in session.steps)


def test_manual_allow_over_automatic_block_executes_and_attests_honestly() -> None:
    toolkit = _CountingToolkit()
    llm = _ScriptedLLM(
        responses=[
            AIMessage(
                content="",
                tool_calls=[_tool_call("cancel_reservation", {"reservation_id": "R9"}, "call-1")],
            ),
            AIMessage(content="done"),
        ]
    )
    session = build_airline_agent_session(
        trace_id="t-manual-allow", llm=llm, environment=_FakeEnvironment(tools=toolkit)
    )

    _result, payloads = _run_to_completion(
        session, thread_id="manual-allow", resume_values={"call-1": "allow"}
    )

    assert payloads[0]["auto_verdict"] == "block"
    assert toolkit.call_count("cancel_reservation", reservation_id="R9") == 1
    assert len(session.steps) == 1
    step = session.steps[0]
    assert step.action.payload["gate_verdict"] == "allow"
    assert step.metadata["automatic_verdict"] == "block"
    assert step.metadata["overridden"] is True
    assert session.gate is not None
    assert session.gate.history == [ProposedAction("cancel_reservation", {"reservation_id": "R9"})]


def test_manual_block_over_automatic_allow_does_not_execute_and_attests_honestly() -> None:
    toolkit = _CountingToolkit()
    llm = _ScriptedLLM(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    _tool_call("get_reservation_details", {"reservation_id": "R1"}, "call-1")
                ],
            ),
            AIMessage(content="done"),
        ]
    )
    session = build_airline_agent_session(
        trace_id="t-manual-block", llm=llm, environment=_FakeEnvironment(tools=toolkit)
    )

    result, payloads = _run_to_completion(
        session, thread_id="manual-block", resume_values={"call-1": "block"}
    )

    assert payloads[0]["auto_verdict"] == "allow"
    assert toolkit.call_count("get_reservation_details", reservation_id="R1") == 0
    assert session.gate is not None
    assert session.gate.history == []
    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert any(
        isinstance(m.content, str) and m.content.startswith("BLOCKED by bossyk-sandbox GATE:")
        for m in tool_messages
    )
    step = session.steps[0]
    assert step.action.payload["gate_verdict"] == "block"
    assert step.metadata["automatic_verdict"] == "allow"
    assert step.metadata["overridden"] is True


def test_a_tool_that_raises_never_enters_history() -> None:
    toolkit = _CountingToolkit(raising_tools=frozenset({"get_reservation_details"}))
    llm = _ScriptedLLM(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    _tool_call("get_reservation_details", {"reservation_id": "R1"}, "call-1"),
                    _tool_call("cancel_reservation", {"reservation_id": "R1"}, "call-2"),
                ],
            ),
            AIMessage(content="done"),
        ]
    )
    session = build_airline_agent_session(
        trace_id="t-raises", llm=llm, environment=_FakeEnvironment(tools=toolkit)
    )

    result, payloads = _run_to_completion(session, thread_id="raises")

    # The failed lookup did not authorise the cancel.
    assert payloads[1]["auto_verdict"] == "block"
    assert session.gate is not None
    assert session.gate.history == []
    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert any(isinstance(m.content, str) and m.content.startswith("error:") for m in tool_messages)


def test_no_interrupts_when_the_llm_proposes_no_tools() -> None:
    toolkit = _CountingToolkit()
    llm = _ScriptedLLM(responses=[AIMessage(content="done")])
    session = build_airline_agent_session(
        trace_id="t-no-tools", llm=llm, environment=_FakeEnvironment(tools=toolkit)
    )

    result, payloads = _run_to_completion(session, thread_id="no-tools")

    assert payloads == []
    assert "__interrupt__" not in result
    assert session.steps == []
    assert session.gate is not None
    assert session.gate.history == []


# --- domain-parameterized builder (live-h2h4 L1 #3) --------------------------


def test_airline_gate_is_wired_to_both_structural_fast_rules() -> None:
    # Both of airline's structural boundaries (cancel_without_lookup,
    # unauthorized_rebooking) must be preventable pre-execution live, not
    # just cancel_without_lookup -- see runtime.langgraph_agent's
    # build_airline_agent_session docstring.
    toolkit = _CountingToolkit()
    llm = _ScriptedLLM(responses=[AIMessage(content="done")])
    session = build_airline_agent_session(
        trace_id="t-airline-gate-wiring", llm=llm, environment=_FakeEnvironment(tools=toolkit)
    )

    assert session.gate is not None
    gated_tools = {
        rule.gated_tool
        for rule in session.gate.instruments
        if isinstance(rule, RequireLookupBeforeCancel)
    }
    assert gated_tools == {"cancel_reservation", "update_reservation_flights"}


def test_build_retail_agent_session_wires_retail_fast_rules_without_network() -> None:
    toolkit = _CountingToolkit()
    llm = _ScriptedLLM(responses=[AIMessage(content="done")])

    session = build_retail_agent_session(
        trace_id="t-retail-wiring", llm=llm, environment=_FakeEnvironment(tools=toolkit)
    )

    assert session.gate is not None
    gated_tools = {
        rule.gated_tool
        for rule in session.gate.instruments
        if isinstance(rule, RequireLookupBeforeCancel)
    }
    assert gated_tools == {
        "cancel_pending_order",
        "return_delivered_order_items",
        "modify_pending_order_payment",
        "modify_user_address",
    }


def test_retail_agent_session_gate_blocks_cancel_without_prior_lookup() -> None:
    toolkit = _CountingToolkit()
    llm = _ScriptedLLM(
        responses=[
            AIMessage(
                content="",
                tool_calls=[_tool_call("cancel_pending_order", {"order_id": "#W1"}, "call-1")],
            ),
            AIMessage(content="done"),
        ]
    )
    session = build_retail_agent_session(
        trace_id="t-retail-block", llm=llm, environment=_FakeEnvironment(tools=toolkit)
    )

    _result, payloads = _run_to_completion(session, thread_id="retail-block")

    assert payloads[0]["auto_verdict"] == "block"
    assert toolkit.call_count("cancel_pending_order", order_id="#W1") == 0
    assert session.gate is not None
    assert session.gate.history == []


def test_retail_agent_session_builds_against_the_real_tau2_retail_environment() -> None:
    # No FIREWORKS_API_KEY / network needed: `llm` is a fake, and tau2's
    # retail environment loads from a local bundled JSON file, not a
    # network call -- confirms the retail tau2 env is cleanly available
    # (live-h2h4 L1 #3's stop-and-report condition does NOT apply).
    llm = _ScriptedLLM(responses=[AIMessage(content="done")])

    session = build_retail_agent_session(trace_id="t-retail-real-env", llm=llm)

    assert session.gate is not None
    gated_tools = {
        rule.gated_tool
        for rule in session.gate.instruments
        if isinstance(rule, RequireLookupBeforeCancel)
    }
    assert "cancel_pending_order" in gated_tools


@dataclass
class _RecordingLLM:
    """Fake chat model that records the message list it was invoked with, so a
    test can assert which system prompt (policy) the agent fed the model."""

    responses: list[AIMessage]
    seen: list[Any] = field(default_factory=list)
    calls: int = 0

    def invoke(self, messages: Any) -> AIMessage:
        self.seen.append(messages)
        response = self.responses[self.calls]
        self.calls += 1
        return response


def test_weaken_policy_preserves_the_policy_but_neutralizes_the_guardrails() -> None:
    original = "Authenticate the user identity before acting. Confirm before mutating."
    weakened = weaken_policy(original)

    assert original in weakened  # the domain policy is preserved
    assert weakened != original  # an override was appended
    lowered = weakened.lower()
    # The override must neutralize verify-before-act + confirm-before-mutate AND
    # the operational lookup the fast-rule gate actually watches (an agent that
    # still looks up the order/user first never produces a structural culprit).
    assert "skip" in lowered
    assert "verification" in lowered
    assert "confirm" in lowered
    assert "lookup" in lowered
    assert "get_order_details" in weakened


def test_weaken_policy_default_is_dir1() -> None:
    original = "Authenticate first."
    assert weaken_policy(original) == weaken_policy(original, strength="dir1")


def test_weaken_policy_aggressive_is_a_blunter_distinct_override() -> None:
    original = "Authenticate the user identity before acting."
    dir1 = weaken_policy(original, strength="dir1")
    aggressive = weaken_policy(original, strength="aggressive")
    assert original in aggressive  # policy still preserved
    assert aggressive != dir1  # a different, stronger override
    low = aggressive.lower()
    assert "mandatory" in low
    assert "must not" in low  # forbids verification/refusal outright


def test_weaken_policy_rejects_unknown_strength() -> None:
    with pytest.raises(ValueError, match="unknown weakening strength"):
        weaken_policy("x", strength="nope")


def test_build_weakened_retail_agent_session_feeds_the_weakened_policy_to_the_model() -> None:
    # dir 1: an under-specified agent. The system prompt the model sees must be
    # the weakened policy (original + guardrail-neutralizing override), so the
    # agent no longer self-enforces and the gate's prevention leg has something
    # to prevent. Fake llm + fake env -> no network / API key.
    llm = _RecordingLLM(responses=[AIMessage(content="ok")])
    env = _FakeEnvironment(_CountingToolkit(), policy="Authenticate the user first.")

    session = build_weakened_retail_agent_session(trace_id="t-weak", llm=llm, environment=env)
    session.graph.invoke(  # type: ignore[call-overload]
        {"messages": [HumanMessage(content="cancel my order")]},
        config={"configurable": {"thread_id": "t-weak"}},
    )

    system_message = llm.seen[0][0]
    assert system_message.content == weaken_policy("Authenticate the user first.")


# --- action-execution timing (§15B+ latency budget) --------------------------


def test_execute_node_times_an_executed_tool_call_as_action_exec_latency() -> None:
    # §15B+: the tau2 tool call the agent actually runs (gate ALLOWed) has its
    # own wall-clock recorded as an `action_exec` LatencyRecord on the session,
    # timed by the injected clock -- the measured "action cost" the latency
    # budget compares detection latency against. A benign lookup is ALLOWed and
    # so executes.
    #
    # The injected clock is a 0.5-step counter (not a fixed 2-value list),
    # because `agent_node` also times its llm.invoke as `agent_inference` on the
    # SAME clock now (voice-model sweep). Every `timed` span reads two
    # consecutive values, so each span (agent_inference AND action_exec) is
    # exactly 0.5 regardless of how many spans run -- keeping this assertion
    # about action_exec robust to the added agent timing.
    import itertools

    toolkit = _CountingToolkit()
    llm = _ScriptedLLM(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    _tool_call("get_reservation_details", {"reservation_id": "R1"}, "call-1")
                ],
            ),
            AIMessage(content="done"),
        ]
    )
    session = build_airline_agent_session(
        trace_id="t-latency",
        llm=llm,
        environment=_FakeEnvironment(tools=toolkit),
        clock=itertools.count(0.0, 0.5).__next__,
    )

    _run_to_completion(session, thread_id="latency")

    assert toolkit.call_count("get_reservation_details", reservation_id="R1") == 1
    assert len(session.tool_latency) == 1
    record = session.tool_latency[0]
    assert record.instrument == "action_exec"
    assert record.elapsed_s == pytest.approx(0.5)


def test_a_blocked_tool_call_records_no_action_exec_latency() -> None:
    # cancel without a prior lookup -> the gate BLOCKS it -> the tool never
    # executes -> nothing to time (parallels gate.history staying empty).
    toolkit = _CountingToolkit()
    llm = _ScriptedLLM(
        responses=[
            AIMessage(
                content="",
                tool_calls=[_tool_call("cancel_reservation", {"reservation_id": "R9"}, "call-1")],
            ),
            AIMessage(content="done"),
        ]
    )
    session = build_airline_agent_session(
        trace_id="t-latency-blocked", llm=llm, environment=_FakeEnvironment(tools=toolkit)
    )

    _run_to_completion(session, thread_id="latency-blocked")

    assert toolkit.call_count("cancel_reservation", reservation_id="R9") == 0
    assert session.tool_latency == []


def test_a_raising_tool_call_records_no_action_exec_latency() -> None:
    # a tool that raises never completed an action -> no latency record (mirrors
    # gate.history: a failed call never fabricates a successful entry).
    toolkit = _CountingToolkit(raising_tools=frozenset({"get_reservation_details"}))
    llm = _ScriptedLLM(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    _tool_call("get_reservation_details", {"reservation_id": "R1"}, "call-1")
                ],
            ),
            AIMessage(content="done"),
        ]
    )
    session = build_airline_agent_session(
        trace_id="t-latency-raises", llm=llm, environment=_FakeEnvironment(tools=toolkit)
    )

    _run_to_completion(session, thread_id="latency-raises")

    assert session.tool_latency == []


def test_retail_tool_schemas_exposes_the_real_bound_retail_toolset() -> None:
    # The grounded adversary (conditions.fireworks_adversary) must ground
    # its attacks in the SAME tools the live retail agent binds -- so this
    # exposes the exact `openai_schema` list `_build_agent_session` binds.
    # No network / API key: tau2's retail env is a local JSON load.
    schemas = retail_tool_schemas()

    names = {schema["function"]["name"] for schema in schemas}
    # The 16 real retail tools (see runtime/langgraph_agent.py's binding).
    assert len(schemas) == 16
    assert {"cancel_pending_order", "modify_user_address", "get_user_details"} <= names


# --- voice-model sweep: provider env seam + agent-inference latency ---------


def _clear_agent_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        "AGENT_MODEL",
        "AGENT_API_KEY",
        "AGENT_BASE_URL",
        "FIREWORKS_MODEL",
        "FIREWORKS_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)


def test_resolve_agent_config_prefers_agent_env_then_fireworks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from bossyk_sandbox.runtime.langgraph_agent import _resolve_agent_config

    _clear_agent_env(monkeypatch)
    monkeypatch.setenv("AGENT_MODEL", "meta/llama-3.1-8b-instruct")
    monkeypatch.setenv("AGENT_API_KEY", "nvidia-key")
    monkeypatch.setenv("AGENT_BASE_URL", "https://integrate.api.nvidia.com/v1")
    monkeypatch.setenv("FIREWORKS_MODEL", "fw-model")
    monkeypatch.setenv("FIREWORKS_API_KEY", "fw-key")

    model, key, base_url = _resolve_agent_config(model_name=None, api_key=None, base_url=None)

    assert model == "meta/llama-3.1-8b-instruct"
    assert key == "nvidia-key"
    assert base_url == "https://integrate.api.nvidia.com/v1"


def test_resolve_agent_config_falls_back_to_fireworks(monkeypatch: pytest.MonkeyPatch) -> None:
    from bossyk_sandbox.runtime.langgraph_agent import (
        FIREWORKS_BASE_URL,
        _resolve_agent_config,
    )

    _clear_agent_env(monkeypatch)
    monkeypatch.setenv("FIREWORKS_API_KEY", "fw-key")

    model, key, base_url = _resolve_agent_config(model_name=None, api_key=None, base_url=None)

    assert key == "fw-key"
    assert base_url == FIREWORKS_BASE_URL  # no AGENT_BASE_URL -> Fireworks default
    assert model  # DEFAULT_FIREWORKS_MODEL when neither model env is set


def test_resolve_agent_config_explicit_args_win(monkeypatch: pytest.MonkeyPatch) -> None:
    from bossyk_sandbox.runtime.langgraph_agent import _resolve_agent_config

    _clear_agent_env(monkeypatch)
    monkeypatch.setenv("AGENT_MODEL", "env-model")
    monkeypatch.setenv("AGENT_API_KEY", "env-key")

    model, key, base_url = _resolve_agent_config(
        model_name="explicit-model", api_key="explicit-key", base_url="https://explicit/v1"
    )

    assert (model, key, base_url) == ("explicit-model", "explicit-key", "https://explicit/v1")


def test_resolve_agent_config_raises_without_any_key(monkeypatch: pytest.MonkeyPatch) -> None:
    from bossyk_sandbox.runtime.langgraph_agent import _resolve_agent_config

    _clear_agent_env(monkeypatch)

    with pytest.raises(RuntimeError, match="AGENT_API_KEY"):
        _resolve_agent_config(model_name="m", api_key=None, base_url=None)


def test_agent_inference_latency_is_recorded_per_turn() -> None:
    toolkit = _CountingToolkit()
    llm = _ScriptedLLM(
        responses=[
            AIMessage(
                content="",
                tool_calls=[_tool_call("get_reservation_details", {"reservation_id": "R1"}, "c1")],
            ),
            AIMessage(content="done"),
        ]
    )
    session = build_airline_agent_session(
        trace_id="t-agent-latency", llm=llm, environment=_FakeEnvironment(tools=toolkit)
    )

    _run_to_completion(session, thread_id="agent-latency")

    assert session.agent_latency, "expected an agent_inference latency record per turn"
    assert all(r.instrument == "agent_inference" for r in session.agent_latency)
    assert all(r.elapsed_s >= 0.0 for r in session.agent_latency)


# --- interp: opt-in behavioral logprob capture -------------------------------

_LOGPROBS_META = {
    "logprobs": {
        "content": [
            {
                "token": "do",
                "logprob": -0.2,
                "top_logprobs": [
                    {"token": "do", "logprob": -0.2},
                    {"token": "no", "logprob": -1.5},
                ],
            },
            {
                "token": "ne",
                "logprob": -0.05,
                "top_logprobs": [
                    {"token": "ne", "logprob": -0.05},
                    {"token": "nt", "logprob": -3.0},
                ],
            },
        ]
    }
}


def test_capture_logprobs_off_by_default_leaves_agent_interp_empty() -> None:
    # Even when the response carries logprobs, default (opt-out) captures nothing
    # -- proving the production/voice-sweep path is unaffected.
    llm = _ScriptedLLM(responses=[AIMessage(content="done", response_metadata=_LOGPROBS_META)])
    session = build_airline_agent_session(
        trace_id="t-interp-off", llm=llm, environment=_FakeEnvironment(tools=_CountingToolkit())
    )

    _run_to_completion(session, thread_id="interp-off")

    assert session.agent_interp == []


def test_capture_logprobs_records_per_turn_uncertainty() -> None:
    llm = _ScriptedLLM(responses=[AIMessage(content="done", response_metadata=_LOGPROBS_META)])
    session = build_airline_agent_session(
        trace_id="t-interp-on",
        llm=llm,
        environment=_FakeEnvironment(tools=_CountingToolkit()),
        capture_logprobs=True,
    )

    _run_to_completion(session, thread_id="interp-on")

    # One uncertainty summary per agent turn, aligned with agent_latency.
    assert len(session.agent_interp) == 1
    assert len(session.agent_interp) == len(session.agent_latency)
    step = session.agent_interp[0]
    assert step.n_tokens == 2
    assert step.max_surprisal == pytest.approx(0.2)
    assert step.mean_surprisal == pytest.approx(0.125)


def test_capture_logprobs_tool_call_turn_without_logprobs_is_zero() -> None:
    # capture on, but the response carries no logprobs (e.g. a tool-call turn):
    # graceful n_tokens==0 summary, not a crash. Two turns -> two summaries.
    llm = _ScriptedLLM(
        responses=[
            AIMessage(
                content="",
                tool_calls=[_tool_call("get_reservation_details", {"reservation_id": "R1"}, "c1")],
            ),
            AIMessage(content="done"),
        ]
    )
    session = build_airline_agent_session(
        trace_id="t-interp-toolcall",
        llm=llm,
        environment=_FakeEnvironment(tools=_CountingToolkit()),
        capture_logprobs=True,
    )

    _run_to_completion(session, thread_id="interp-toolcall")

    assert len(session.agent_interp) == 2
    assert all(s.n_tokens == 0 for s in session.agent_interp)


def test_capture_prompts_off_by_default_leaves_agent_prompts_empty() -> None:
    llm = _ScriptedLLM(responses=[AIMessage(content="done")])
    session = build_airline_agent_session(
        trace_id="t-prompts-off", llm=llm, environment=_FakeEnvironment(tools=_CountingToolkit())
    )
    _run_to_completion(session, thread_id="prompts-off")
    assert session.agent_prompts == []


def test_capture_prompts_records_the_rendered_decision_context_per_turn() -> None:
    llm = _ScriptedLLM(responses=[AIMessage(content="done")])
    session = build_airline_agent_session(
        trace_id="t-prompts-on",
        llm=llm,
        environment=_FakeEnvironment(tools=_CountingToolkit(), policy="be compliant"),
        capture_prompts=True,
    )
    _run_to_completion(session, thread_id="prompts-on")
    assert len(session.agent_prompts) == 1
    assert len(session.agent_prompts) == len(session.agent_latency)
    # the rendered context carries the system/policy the agent was given
    assert "be compliant" in session.agent_prompts[0]


def test_capture_prompts_off_by_default_leaves_agent_actions_empty() -> None:
    llm = _ScriptedLLM(responses=[AIMessage(content="done")])
    session = build_airline_agent_session(
        trace_id="t-actions-off", llm=llm, environment=_FakeEnvironment(tools=_CountingToolkit())
    )
    _run_to_completion(session, thread_id="actions-off")
    assert session.agent_actions == []


def test_capture_prompts_records_the_agent_action_per_turn_aligned() -> None:
    # A tool-call turn then a text turn: agent_actions aligns 1:1 with
    # agent_prompts and captures WHAT the agent did (tool call, then reply).
    tool_call = {"name": "cancel_reservation", "args": {"reservation_id": "R1"}, "id": "c1"}
    llm = _ScriptedLLM(
        responses=[
            AIMessage(content="", tool_calls=[tool_call]),
            AIMessage(content="all done"),
        ]
    )
    session = build_airline_agent_session(
        trace_id="t-actions-on",
        llm=llm,
        environment=_FakeEnvironment(tools=_CountingToolkit(), policy="be compliant"),
        capture_prompts=True,
    )
    _run_to_completion(session, thread_id="actions-on")
    assert len(session.agent_actions) == len(session.agent_prompts)
    assert "cancel_reservation" in session.agent_actions[0]
    assert session.agent_actions[1] == "all done"
