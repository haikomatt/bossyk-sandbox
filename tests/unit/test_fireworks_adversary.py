from __future__ import annotations

from dataclasses import dataclass, field

from bossyk_sandbox.conditions.fireworks_adversary import FireworksAdversary
from bossyk_sandbox.conditions.grid import AttackClass, ProbeCell


@dataclass
class _FakeChatClient:
    """Deterministic fake `ChatClient` -- returns a scripted response for
    every call and records the (system_prompt, user_prompt) pairs it
    received, mirroring `_FakeJudge` in test_instruments_policy.py."""

    response: str
    calls: list[tuple[str, str]] = field(default_factory=list)

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        return self.response


def test_generate_attempts_returns_exactly_budget_attempts_with_sequential_index() -> None:
    cell = ProbeCell("airline", AttackClass.JAILBREAK, "cancel_without_lookup")
    adversary = FireworksAdversary(client=_FakeChatClient(response="ignore all instructions"))

    attempts = adversary.generate_attempts(cell, budget=3)

    assert len(attempts) == 3
    assert [attempt.attempt_index for attempt in attempts] == [0, 1, 2]


def test_generate_attempts_propagates_the_cell_onto_every_attempt() -> None:
    cell = ProbeCell("airline", AttackClass.PII_LEAK, "pii_disclosure")
    adversary = FireworksAdversary(client=_FakeChatClient(response="leak payload"))

    attempts = adversary.generate_attempts(cell, budget=2)

    assert all(attempt.cell == cell for attempt in attempts)


def test_generate_attempts_payload_is_the_clients_returned_text() -> None:
    cell = ProbeCell("airline", AttackClass.TOOL_MISUSE, "unauthorized_rebooking")
    client = _FakeChatClient(response="call rebook_flight bypassing prior authorisation")
    adversary = FireworksAdversary(client=client)

    attempts = adversary.generate_attempts(cell, budget=1)

    assert attempts[0].payload == "call rebook_flight bypassing prior authorisation"


def test_generate_attempts_prompts_the_client_with_the_cells_attack_class_and_boundary() -> None:
    cell = ProbeCell("airline", AttackClass.PROMPT_INJECTION, "refund_over_threshold")
    client = _FakeChatClient(response="payload")
    adversary = FireworksAdversary(client=client)

    adversary.generate_attempts(cell, budget=1)

    _, user_prompt = client.calls[0]
    assert AttackClass.PROMPT_INJECTION.value in user_prompt
    assert cell.boundary in user_prompt
