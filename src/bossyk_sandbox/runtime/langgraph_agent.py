from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Annotated, Any, TypedDict

from auditk.schema import Step
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt
from pydantic import SecretStr
from tau2.domains.airline.environment import get_environment as get_airline_environment
from tau2.domains.retail.environment import get_environment as get_retail_environment

from bossyk_sandbox.evidence.trace import make_attested_step
from bossyk_sandbox.gate import Gate
from bossyk_sandbox.instruments.base import Decision, Instrument, ProposedAction, Verdict
from bossyk_sandbox.scenarios.runner import default_fast_rules, retail_fast_rules

# Fireworks exposes an OpenAI-compatible endpoint, so the same ChatOpenAI
# client used elsewhere in this codebase (e.g. no separate SDK) works here
# too — just point base_url/api_key at Fireworks. Convention matches
# auditk-constellaration-experiment's .env: FIREWORKS_API_KEY + FIREWORKS_MODEL.
# firefunction-v2 (the original Phase 0 pick) was retired from Fireworks'
# serverless catalog. deepseek-v4-pro was tried next but rejected: this
# codebase's judge path also runs on a deepseek model, and using the same
# model family for the agent-under-test and its judge/evaluator violates
# the same-family self-evaluation independence this project is built
# against. kimi-k2p6 is a different family with solid tool-calling support.
FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"
DEFAULT_FIREWORKS_MODEL = "accounts/fireworks/models/kimi-k2p6"


class AgentState(TypedDict):
    messages: Annotated[list[Any], add_messages]
    pending_calls: list[dict[str, Any]]
    active_call: dict[str, Any] | None
    auto_verdict: str | None
    auto_reason: str | None
    final_verdict: str | None


@dataclass
class AirlineAgentSession:
    """A compiled live agent plus the running list of GATE-scored Steps it
    has produced so far. `steps` is appended to in place as the graph runs,
    so it can be handed to `evidence.trace.build_trace` once the session
    (or the exit demo) is done.

    Despite the name (kept for backward compatibility with existing
    imports/tests), nothing about this dataclass is airline-specific --
    `_build_agent_session` returns the same shape for
    `build_retail_agent_session` too. `AgentSession` is the domain-neutral
    alias for new code.
    """

    graph: CompiledStateGraph[AgentState, Any, AgentState, AgentState]
    trace_id: str
    steps: list[Step] = field(default_factory=list)
    gate: Gate | None = None


AgentSession = AirlineAgentSession


def _tool_schemas(toolkit: Any) -> list[dict[str, Any]]:
    return [tool.openai_schema for tool in toolkit.get_tools().values()]


def retail_tool_schemas() -> list[dict[str, Any]]:
    """The 16 real tau2 retail tool schemas the live retail agent binds
    (`build_retail_agent_session`), exposed so the grounded adversary
    (conditions.fireworks_adversary) grounds attacks in the agent's actual
    toolset. Local tau2 env load -- no network / API key."""
    return _tool_schemas(get_retail_environment().tools)


def _build_agent_session(
    *,
    trace_id: str,
    get_environment_fn: Callable[[], Any],
    fast_rules: list[Instrument],
    model_name: str | None,
    api_key: str | None,
    base_url: str,
    llm: Any | None,
    environment: Any | None,
) -> AgentSession:
    """Domain-parameterized live LangGraph agent with in-graph tool-call
    interception, shared by `build_airline_agent_session` and
    `build_retail_agent_session`.

    Every proposed tool call is scored by the same `Gate`/`Instrument`
    machinery as the stub agent (bossyk_sandbox.gate), then surfaced via
    `interrupt()` so a human (via the console) can confirm or override the
    automatic verdict before the tau2 tool actually runs. Because a single
    AIMessage can carry several tool calls but LangGraph re-runs a node from
    the top on every resume (only the `interrupt()` value itself is cached),
    scoring/deciding/executing are split across separate nodes
    (`plan_calls` -> `prepare` -> `approve` -> `execute`, looping back to
    `prepare` while calls remain) so no non-idempotent work — scoring,
    executing the tau2 tool, recording history — shares a node with
    `interrupt()`. Each scored call is also recorded as a spec-conformant
    Step in `session.steps`, mirroring the stub agent's trace path.

    Defaults to Fireworks' OpenAI-compatible endpoint: `model_name` falls
    back to $FIREWORKS_MODEL (then `firefunction-v2`), `api_key` falls back
    to $FIREWORKS_API_KEY. Pass `base_url`/`api_key`/`model_name` explicitly
    to target a different OpenAI-compatible provider instead. Pass `llm`
    to use an already-constructed chat model (e.g. a test double) instead,
    skipping the API-key check entirely; pass `environment` to use an
    already-resolved tau2 environment instead of calling `get_environment_fn()`.
    """
    env = environment if environment is not None else get_environment_fn()
    toolkit = env.tools

    if llm is None:
        resolved_model = model_name or os.environ.get("FIREWORKS_MODEL", DEFAULT_FIREWORKS_MODEL)
        resolved_api_key = api_key or os.environ.get("FIREWORKS_API_KEY")
        if not resolved_api_key:
            raise RuntimeError(
                "FIREWORKS_API_KEY is required to run the live agent "
                "(set it in .env, matching auditk-constellaration-experiment's convention)."
            )
        tool_schemas = _tool_schemas(toolkit)
        llm = ChatOpenAI(
            model=resolved_model,
            base_url=base_url,
            api_key=SecretStr(resolved_api_key),
            temperature=0,
        ).bind_tools(tool_schemas)

    gate = Gate(instruments=fast_rules)
    steps: list[Step] = []

    def agent_node(state: AgentState) -> dict[str, Any]:
        messages = [SystemMessage(content=env.policy), *state["messages"]]
        response = llm.invoke(messages)
        return {"messages": [response]}

    def plan_calls_node(state: AgentState) -> dict[str, Any]:
        last = state["messages"][-1]
        tool_calls = last.tool_calls if isinstance(last, AIMessage) else []
        return {"pending_calls": list(tool_calls)}

    def prepare_node(state: AgentState) -> dict[str, Any]:
        active = state["pending_calls"][0]
        proposed = ProposedAction(active["name"], dict(active["args"]))
        decision = gate.score(proposed)
        return {
            "active_call": active,
            "auto_verdict": decision.verdict.value,
            "auto_reason": decision.reason,
        }

    def approve_node(state: AgentState) -> dict[str, Any]:
        call = state["active_call"]
        auto_verdict = state["auto_verdict"]
        assert call is not None
        assert auto_verdict is not None
        override = interrupt(
            {
                "tool_call_id": call["id"],
                "tool_name": call["name"],
                "arguments": call["args"],
                "auto_verdict": auto_verdict,
                "auto_reason": state["auto_reason"],
            }
        )
        final = Verdict(override) if override else Verdict(auto_verdict)
        return {"final_verdict": final.value}

    def execute_node(state: AgentState) -> dict[str, Any]:
        call = state["active_call"]
        assert call is not None
        proposed = ProposedAction(call["name"], dict(call["args"]))
        auto_verdict = state["auto_verdict"]
        assert auto_verdict is not None
        auto_decision = Decision(Verdict(auto_verdict), state["auto_reason"] or "")
        final_verdict_str = state["final_verdict"]
        assert final_verdict_str is not None
        final = Verdict(final_verdict_str)

        step = make_attested_step(
            trace_id=trace_id, proposed=proposed, auto_decision=auto_decision, final_verdict=final
        )
        steps.append(step)

        if final is Verdict.ALLOW:
            try:
                result = toolkit.use_tool(call["name"], **call["args"])
            except Exception as exc:  # tau2 tool raised — surface as an error, not a crash
                tool_message = ToolMessage(content=f"error: {exc}", tool_call_id=call["id"])
            else:
                gate.record(proposed)
                tool_message = ToolMessage(content=str(result), tool_call_id=call["id"])
        else:
            gate_reason = step.action.payload["gate_reason"]
            tool_message = ToolMessage(
                content=f"BLOCKED by bossyk-sandbox GATE: {gate_reason}",
                tool_call_id=call["id"],
            )

        return {
            "messages": [tool_message],
            "pending_calls": state["pending_calls"][1:],
            "active_call": None,
            "auto_verdict": None,
            "auto_reason": None,
            "final_verdict": None,
        }

    def route_after_agent(state: AgentState) -> str:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "plan_calls"
        return END

    def route_after_execute(state: AgentState) -> str:
        return "prepare" if state["pending_calls"] else "agent"

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("plan_calls", plan_calls_node)
    graph.add_node("prepare", prepare_node)
    graph.add_node("approve", approve_node)
    graph.add_node("execute", execute_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", route_after_agent, {"plan_calls": "plan_calls", END: END})
    graph.add_edge("plan_calls", "prepare")
    graph.add_edge("prepare", "approve")
    graph.add_edge("approve", "execute")
    graph.add_conditional_edges(
        "execute", route_after_execute, {"prepare": "prepare", "agent": "agent"}
    )

    compiled = graph.compile(checkpointer=MemorySaver())
    return AgentSession(graph=compiled, trace_id=trace_id, steps=steps, gate=gate)


def build_airline_agent_session(
    *,
    trace_id: str = "live-airline-session",
    model_name: str | None = None,
    api_key: str | None = None,
    base_url: str = FIREWORKS_BASE_URL,
    llm: Any | None = None,
    environment: Any | None = None,
) -> AgentSession:
    """Live LangGraph airline agent. See `_build_agent_session` for the
    shared mechanics. Fast-path gate wired to `default_fast_rules()`
    (scenarios/runner.py): cancel_reservation AND update_reservation_flights,
    both gated on a prior get_reservation_details lookup -- so both of
    airline's structural consequence boundaries (cancel_without_lookup,
    unauthorized_rebooking) are preventable pre-execution live, not just
    cancel_without_lookup."""
    return _build_agent_session(
        trace_id=trace_id,
        get_environment_fn=get_airline_environment,
        fast_rules=default_fast_rules(),
        model_name=model_name,
        api_key=api_key,
        base_url=base_url,
        llm=llm,
        environment=environment,
    )


def build_retail_agent_session(
    *,
    trace_id: str = "live-retail-session",
    model_name: str | None = None,
    api_key: str | None = None,
    base_url: str = FIREWORKS_BASE_URL,
    llm: Any | None = None,
    environment: Any | None = None,
) -> AgentSession:
    """Live LangGraph retail agent -- the retail counterpart of
    `build_airline_agent_session`, needed so retail crossings are reachable
    live (retail-primary: the first billable L1 run targets retail). Same
    mechanics via `_build_agent_session`, pointed at tau2's retail
    environment and `retail_fast_rules()` (cancel_pending_order,
    return_delivered_order_items, modify_pending_order_payment,
    modify_user_address, each gated on a prior lookup)."""
    return _build_agent_session(
        trace_id=trace_id,
        get_environment_fn=get_retail_environment,
        fast_rules=retail_fast_rules(),
        model_name=model_name,
        api_key=api_key,
        base_url=base_url,
        llm=llm,
        environment=environment,
    )
