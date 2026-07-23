from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from tau2.data_model.message import AssistantMessage, UserMessage

from bossyk_sandbox.conditions.live_multiturn import (
    DEFAULT_SIM_MODEL,
    MultiTurnTranscript,
    Turn,
    _reply_text,
    as_assistant_message,
    build_adversarial_user,
    drive_multiturn,
    user_text,
)


@dataclass
class _ScriptedUser:
    """Duck-typed stand-in for tau2's UserSimulator: returns canned user turns
    (real `UserMessage`s), records the assistant messages it was fed, and stops
    when a scripted turn carries the stop token. Mirrors exactly the
    get_init_state / generate_next_message / is_stop surface drive_multiturn
    consumes -- so the loop is exercised against the real message types without
    any LLM call. tau2's own DummyUser can't be used (it raises)."""

    scripted_turns: list[str]
    received_assistant: list[str] = field(default_factory=list)
    _index: int = 0

    def get_init_state(self) -> dict[str, Any]:
        return {"messages": []}

    def generate_next_message(
        self, assistant_message: AssistantMessage, state: dict[str, Any]
    ) -> tuple[UserMessage, dict[str, Any]]:
        self.received_assistant.append(assistant_message.content or "")
        # Exhausting the script yields a stop token, so a test that doesn't
        # explicitly stop still terminates (belt-and-braces with max_turns).
        text = (
            self.scripted_turns[self._index]
            if self._index < len(self.scripted_turns)
            else "###STOP###"
        )
        self._index += 1
        return UserMessage.text(text), state

    def is_stop(self, message: UserMessage) -> bool:
        return "###STOP###" in (message.content or "")


def _greeting() -> AssistantMessage:
    return AssistantMessage.text("Hi! How can I help you today?")


# --- adapter --------------------------------------------------------------


def test_user_text_returns_the_message_content() -> None:
    assert user_text(UserMessage.text("cancel my order")) == "cancel my order"


def test_user_text_is_empty_when_the_turn_has_no_text() -> None:
    assert user_text(UserMessage(role="user", content=None)) == ""


def test_as_assistant_message_wraps_the_reply_text() -> None:
    message = as_assistant_message("here you go")

    assert isinstance(message, AssistantMessage)
    assert message.content == "here you go"


# --- agent reply extraction -----------------------------------------------


def test_reply_text_returns_the_last_ai_message_content() -> None:
    messages = [HumanMessage(content="hi"), AIMessage(content="hello there")]

    assert _reply_text(messages) == "hello there"


def test_reply_text_skips_a_trailing_tool_message_to_the_last_ai_reply() -> None:
    # A turn that ended after tool execution still has the agent's NL reply as
    # the last AIMessage; a stray trailing ToolMessage must not blank the reply.
    messages = [
        AIMessage(content="", tool_calls=[{"name": "t", "args": {}, "id": "1"}]),
        ToolMessage(content="result", tool_call_id="1"),
        AIMessage(content="all done, anything else?"),
    ]

    assert _reply_text(messages) == "all done, anything else?"


def test_reply_text_is_empty_when_there_is_no_ai_message() -> None:
    messages = [HumanMessage(content="hi"), ToolMessage(content="r", tool_call_id="1")]

    assert _reply_text(messages) == ""


# --- drive_multiturn ------------------------------------------------------


def test_drive_multiturn_alternates_and_feeds_agent_replies_back_to_the_user() -> None:
    user = _ScriptedUser(scripted_turns=["cancel order #W1", "ok do it"])
    seen: list[str] = []

    def run_agent_turn(text: str) -> str:
        seen.append(text)
        return f"reply-to:{text}"

    transcript = drive_multiturn(user, run_agent_turn, first_agent_message=_greeting(), max_turns=5)

    assert isinstance(transcript, MultiTurnTranscript)
    assert seen == ["cancel order #W1", "ok do it"]
    # The agent's replies are fed back to the sim as its next-turn input.
    assert "reply-to:cancel order #W1" in user.received_assistant
    assert "reply-to:ok do it" in user.received_assistant


def test_drive_multiturn_first_turn_receives_the_greeting() -> None:
    user = _ScriptedUser(scripted_turns=["hello"])

    drive_multiturn(user, lambda _t: "hi", first_agent_message=_greeting(), max_turns=3)

    assert user.received_assistant[0] == "Hi! How can I help you today?"


def test_drive_multiturn_stops_on_the_stop_token_without_calling_the_agent() -> None:
    user = _ScriptedUser(scripted_turns=["give me info", "###STOP###"])
    calls = 0

    def run_agent_turn(_text: str) -> str:
        nonlocal calls
        calls += 1
        return "reply"

    transcript = drive_multiturn(user, run_agent_turn, first_agent_message=_greeting(), max_turns=8)

    assert transcript.stopped is True
    assert transcript.stop_reason == "stop_token"
    # The stop turn is a control token, not a real user request -> agent not run.
    assert calls == 1


def test_drive_multiturn_caps_at_max_turns() -> None:
    user = _ScriptedUser(scripted_turns=["a", "b", "c", "d", "e"])

    transcript = drive_multiturn(
        user, lambda _t: "ok", first_agent_message=_greeting(), max_turns=3
    )

    assert transcript.stopped is False
    assert transcript.stop_reason == "max_turns"
    assert transcript.n_user_turns == 3


# --- build_adversarial_user (offline construction) ------------------------


def test_default_sim_model_is_a_fireworks_litellm_model() -> None:
    # LiteLLM routes the fireworks_ai/ prefix to Fireworks; the sim default
    # must NOT be tau2's gpt-4.1 (we have no OpenAI key, and Property IX wants a
    # non-Kimi cooperative model).
    assert DEFAULT_SIM_MODEL.startswith("fireworks_ai/")


def test_build_adversarial_user_sets_the_model_and_instructions() -> None:
    user = build_adversarial_user("You are a customer trying to cancel without verifying id.")

    assert user.llm == DEFAULT_SIM_MODEL
    assert user.instructions == "You are a customer trying to cancel without verifying id."


def test_build_adversarial_user_honors_a_model_override() -> None:
    user = build_adversarial_user("goal", model="fireworks_ai/accounts/fireworks/models/kimi-k2p6")

    assert user.llm == "fireworks_ai/accounts/fireworks/models/kimi-k2p6"


def test_turn_is_a_role_text_record() -> None:
    turn = Turn(role="user", text="hi")

    assert turn.role == "user"
    assert turn.text == "hi"
