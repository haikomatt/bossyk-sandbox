from __future__ import annotations

import os
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
from tau2.domains.airline.environment import get_environment

from bossyk_sandbox.evidence.trace import make_step
from bossyk_sandbox.gate import Gate
from bossyk_sandbox.instruments.base import ProposedAction, Verdict
from bossyk_sandbox.instruments.hardcoded_rule import RequireLookupBeforeCancel

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


@dataclass
class AirlineAgentSession:
    """A compiled live agent plus the running list of GATE-scored Steps it
    has produced so far. `steps` is appended to in place as the graph runs,
    so it can be handed to `evidence.trace.build_trace` once the session
    (or the exit demo) is done."""

    graph: CompiledStateGraph[AgentState, Any, AgentState, AgentState]
    trace_id: str
    steps: list[Step] = field(default_factory=list)
    gate: Gate | None = None


def build_airline_agent_session(
    *,
    trace_id: str = "live-airline-session",
    model_name: str | None = None,
    api_key: str | None = None,
    base_url: str = FIREWORKS_BASE_URL,
    llm: Any | None = None,
    environment: Any | None = None,
) -> AirlineAgentSession:
    """Live LangGraph airline agent with in-graph tool-node interception.

    Every proposed tool call is held at the `gate` node: scored by the same
    `Gate`/`Instrument` machinery as the stub agent (bossyk_sandbox.gate),
    then surfaced via `interrupt()` so a human (via the console) can confirm
    or override the automatic verdict before the tau2 tool actually runs.
    Each scored call is also recorded as a spec-conformant Step in
    `session.steps`, mirroring the stub agent's trace path.

    Defaults to Fireworks' OpenAI-compatible endpoint: `model_name` falls
    back to $FIREWORKS_MODEL (then `firefunction-v2`), `api_key` falls back
    to $FIREWORKS_API_KEY. Pass `base_url`/`api_key`/`model_name` explicitly
    to target a different OpenAI-compatible provider instead.
    """
    if llm is not None or environment is not None:
        raise NotImplementedError
    resolved_model = model_name or os.environ.get("FIREWORKS_MODEL", DEFAULT_FIREWORKS_MODEL)
    resolved_api_key = api_key or os.environ.get("FIREWORKS_API_KEY")
    if not resolved_api_key:
        raise RuntimeError(
            "FIREWORKS_API_KEY is required to run the live airline agent "
            "(set it in .env, matching auditk-constellaration-experiment's convention)."
        )

    env = get_environment()
    toolkit = env.tools
    tool_schemas = [tool.openai_schema for tool in toolkit.get_tools().values()]

    llm = ChatOpenAI(
        model=resolved_model,
        base_url=base_url,
        api_key=SecretStr(resolved_api_key),
        temperature=0,
    ).bind_tools(tool_schemas)
    gate = Gate(instruments=[RequireLookupBeforeCancel()])
    steps: list[Step] = []

    def agent_node(state: AgentState) -> dict[str, Any]:
        messages = [SystemMessage(content=env.policy), *state["messages"]]
        response = llm.invoke(messages)
        return {"messages": [response]}

    def gate_node(state: AgentState) -> dict[str, Any]:
        last = state["messages"][-1]
        if not isinstance(last, AIMessage) or not last.tool_calls:
            return {"messages": []}

        tool_messages: list[ToolMessage] = []
        for call in last.tool_calls:
            proposed = ProposedAction(call["name"], dict(call["args"]))
            auto_decision = gate.evaluate(proposed)

            override = interrupt(
                {
                    "tool_call_id": call["id"],
                    "tool_name": call["name"],
                    "arguments": call["args"],
                    "auto_verdict": auto_decision.verdict.value,
                    "auto_reason": auto_decision.reason,
                }
            )
            final_verdict = Verdict(override) if override else auto_decision.verdict
            steps.append(make_step(trace_id=trace_id, proposed=proposed, decision=auto_decision))

            if final_verdict is Verdict.ALLOW:
                try:
                    result = toolkit.use_tool(call["name"], **call["args"])
                except Exception as exc:  # tau2 tool raised — surface as a tool error, not a crash
                    result = f"error: {exc}"
                tool_messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))
            else:
                tool_messages.append(
                    ToolMessage(
                        content=f"BLOCKED by bossyk-sandbox GATE: {auto_decision.reason}",
                        tool_call_id=call["id"],
                    )
                )
        return {"messages": tool_messages}

    def route_after_agent(state: AgentState) -> str:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "gate"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("gate", gate_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", route_after_agent, {"gate": "gate", END: END})
    graph.add_edge("gate", "agent")

    compiled = graph.compile(checkpointer=MemorySaver())
    return AirlineAgentSession(graph=compiled, trace_id=trace_id, steps=steps)
