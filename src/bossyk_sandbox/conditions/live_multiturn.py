from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command
from tau2.data_model.message import AssistantMessage, UserMessage
from tau2.user.user_simulator import UserSimulator

from bossyk_sandbox.conditions.live_replay import LiveRunResult, _live_run_result
from bossyk_sandbox.runtime.langgraph_agent import AgentSession, build_retail_agent_session

# Path B (phase-live-h2h4-path-b-usersim.md): a multi-turn adversarial user
# (tau2 `UserSimulator`, driven standalone -- NOT tau2's Orchestrator) converses
# with our live langgraph agent over several turns, supplying the follow-ups
# (identity, confirmation, insistence) a single-turn attack can't. The whole
# episode presents as the same `run_session(str) -> LiveSessionResult` that
# `conditions.live_replay.replay_crossing` already consumes, so the boundary
# oracle + post-hoc policy + live-H2/H4 math reuse unchanged.
#
# `run_live_multiturn_retail_session` is the ONLY network-touching, billable
# function here (real Fireworks agent + sim calls); it must never run in the
# deterministic suite -- only scripts/live_h2h4_bench.py calls it (gated).
# `drive_multiturn` + the adapter + `_reply_text` are pure and fully tested with
# a scripted fake user + fake agent-turn.

GREETING = "Hi! How can I help you today?"
DEFAULT_SIM_MODEL = "fireworks_ai/accounts/fireworks/models/deepseek-v4-pro"


@dataclass(frozen=True)
class Turn:
    """One conversation turn: role in {"user", "agent"} and its text."""

    role: str
    text: str


@dataclass(frozen=True)
class MultiTurnTranscript:
    """The record of one multi-turn episode -- the ordered turns plus how it
    ended (the sim's stop token vs the max_turns cap). Proposed/executed calls
    are read off the agent session separately (into a `LiveRunResult`); this is
    the conversational record for observability + testing."""

    turns: list[Turn]
    n_user_turns: int
    stopped: bool
    stop_reason: str


def user_text(message: UserMessage) -> str:
    """Plain text of a tau2 user turn to feed our langchain agent; '' if the
    turn carries no text (e.g. a pure tool call)."""
    return message.content or ""


def as_assistant_message(reply: str) -> AssistantMessage:
    """Wrap our agent's NL reply as the `AssistantMessage` the sim expects as
    its next-turn input."""
    return AssistantMessage.text(reply)


def _reply_text(messages: list[Any]) -> str:
    """The agent's natural-language reply from a completed turn: the content of
    the last `AIMessage`. Returns '' if the turn produced no text AIMessage
    (e.g. it ended on a ToolMessage) -- an empty reply is honest, not a crash,
    and the sim can still respond to it."""
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            content = message.content
            if isinstance(content, str):
                return content
    return ""


def drive_multiturn(
    user: Any,
    run_agent_turn: Callable[[str], str],
    *,
    first_agent_message: AssistantMessage,
    max_turns: int,
) -> MultiTurnTranscript:
    """Alternate the adversarial `user` (anything with the tau2
    get_init_state / generate_next_message / is_stop surface) and our agent
    (`run_agent_turn(user_text) -> reply`): seed the greeting, get the user's
    turn, run the agent, feed its reply back, repeat -- until the sim emits a
    stop token or `max_turns` user turns elapse. Both `user` and
    `run_agent_turn` are injected, so this is pure and network-free in tests."""
    state = user.get_init_state()
    assistant_message = first_agent_message
    turns: list[Turn] = []
    stopped = False
    for _ in range(max_turns):
        user_msg, state = user.generate_next_message(assistant_message, state)
        text = user_text(user_msg)
        turns.append(Turn(role="user", text=text))
        if user.is_stop(user_msg):
            stopped = True
            break
        reply = run_agent_turn(text)
        turns.append(Turn(role="agent", text=reply))
        assistant_message = as_assistant_message(reply)
    n_user_turns = sum(1 for turn in turns if turn.role == "user")
    stop_reason = "stop_token" if stopped else "max_turns"
    return MultiTurnTranscript(
        turns=turns, n_user_turns=n_user_turns, stopped=stopped, stop_reason=stop_reason
    )


def build_adversarial_user(
    instructions: str, *, model: str = DEFAULT_SIM_MODEL, temperature: float = 1.0
) -> UserSimulator:
    """A tau2 `UserSimulator` whose free-text `instructions` are the adversarial
    goal, on a Fireworks litellm model (default deepseek -- non-Kimi, Property
    IX; tau2's gpt-4.1 default needs an OpenAI key we don't have). Offline to
    construct -- litellm is only called at `generate_next_message` time -- so no
    network/key is needed until the episode actually runs."""
    return UserSimulator(
        llm=model, instructions=instructions, llm_args={"temperature": temperature}
    )


def _drive_agent_turn(session: AgentSession, user_message: str, trace_id: str) -> str:
    """One agent turn on the SAME session/thread (state accumulates across
    turns via MemorySaver): feed the user message, drive every interrupt to
    completion (mirrors live_replay._drive_session), return the agent's NL
    reply. Billable -- real agent calls."""
    config = {"configurable": {"thread_id": trace_id}}
    result = session.graph.invoke(  # type: ignore[call-overload]
        {"messages": [HumanMessage(content=user_message)]}, config=config
    )
    while "__interrupt__" in result:
        interrupt_payload = result["__interrupt__"][0].value
        result = session.graph.invoke(  # type: ignore[call-overload]
            Command(resume=interrupt_payload["auto_verdict"]), config=config
        )
    return _reply_text(result["messages"])


def run_live_multiturn_retail_session(
    instructions: str, *, max_turns: int = 8, model: str = DEFAULT_SIM_MODEL
) -> LiveRunResult:
    """REAL, network-touching, billable runner: builds one live retail agent
    session + a Fireworks adversarial `UserSimulator` with `instructions`, runs
    a multi-turn episode, and returns the session's accumulated
    proposed/executed history (reusing `live_replay._live_run_result`). Never
    call from the deterministic suite -- only scripts/live_h2h4_bench.py does,
    gated behind RUN_LIVE_H2_E2E=1."""
    trace_id = f"live-h2-retail-mt-{uuid.uuid4()}"
    session = build_retail_agent_session(trace_id=trace_id)
    user = build_adversarial_user(instructions, model=model)

    def run_agent_turn(user_message: str) -> str:
        return _drive_agent_turn(session, user_message, trace_id)

    drive_multiturn(
        user,
        run_agent_turn,
        first_agent_message=as_assistant_message(GREETING),
        max_turns=max_turns,
    )
    return _live_run_result(session, trace_id, agent_config_ref="live-h2h4-retail-multiturn@0.1")
