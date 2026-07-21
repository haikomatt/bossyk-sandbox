from __future__ import annotations

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
from tau2.domains.airline.environment import get_environment

from bossyk_sandbox.evidence.trace import make_step
from bossyk_sandbox.gate import Gate
from bossyk_sandbox.instruments.base import ProposedAction, Verdict
from bossyk_sandbox.instruments.hardcoded_rule import RequireLookupBeforeCancel


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


def build_airline_agent_session(
    *, trace_id: str = "live-airline-session", model_name: str = "gpt-4o-mini"
) -> AirlineAgentSession:
    """Live LangGraph airline agent with in-graph tool-node interception.

    Every proposed tool call is held at the `gate` node: scored by the same
    `Gate`/`Instrument` machinery as the stub agent (bossyk_sandbox.gate),
    then surfaced via `interrupt()` so a human (via the console) can confirm
    or override the automatic verdict before the tau2 tool actually runs.
    Each scored call is also recorded as a spec-conformant Step in
    `session.steps`, mirroring the stub agent's trace path.
    """
    env = get_environment()
    toolkit = env.tools
    tool_schemas = [tool.openai_schema for tool in toolkit.get_tools().values()]

    llm = ChatOpenAI(model=model_name, temperature=0).bind_tools(tool_schemas)
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
