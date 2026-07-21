from __future__ import annotations

from dataclasses import dataclass

from bossyk_sandbox.instruments.base import ProposedAction
from bossyk_sandbox.instruments.policy import PolicyInstrument


@dataclass
class _FakeStepResult:
    label: str
    reasoning: str


class _FakeJudge:
    """A fake matching bossyk's `PolicyAwareJudge.score_step` signature —
    deterministic, no Fireworks call."""

    def __init__(self, label: str, reasoning: str) -> None:
        self._label = label
        self._reasoning = reasoning
        self.calls: list[tuple[str, str, str | None]] = []

    def score_step(
        self, step_id: str, action_text: str, declared_intent: str | None = None
    ) -> _FakeStepResult:
        self.calls.append((step_id, action_text, declared_intent))
        return _FakeStepResult(label=self._label, reasoning=self._reasoning)


def test_policy_instrument_reports_faithful_label() -> None:
    judge = _FakeJudge("faithful", "within authorised goals")
    instrument = PolicyInstrument(judge=judge)
    proposed = ProposedAction(
        "get_reservation_details", {"reservation_id": "R1"}, declared_intent="look up reservation"
    )

    verdict = instrument.annotate(proposed, history=[])

    assert verdict.instrument == "policy"
    assert verdict.label == "faithful"
    assert verdict.detail == "within authorised goals"


def test_policy_instrument_reports_violation_label() -> None:
    judge = _FakeJudge("instruction_noncompliance", "cancels without prior lookup")
    instrument = PolicyInstrument(judge=judge)
    proposed = ProposedAction("cancel_reservation", {"reservation_id": "R1"})

    verdict = instrument.annotate(proposed, history=[])

    assert verdict.label == "instruction_noncompliance"


def test_policy_instrument_assigns_unique_step_ids_across_calls() -> None:
    judge = _FakeJudge("faithful", "ok")
    instrument = PolicyInstrument(judge=judge)

    instrument.annotate(
        ProposedAction("get_reservation_details", {"reservation_id": "R1"}), history=[]
    )
    instrument.annotate(ProposedAction("cancel_reservation", {"reservation_id": "R1"}), history=[])

    step_ids = [call[0] for call in judge.calls]
    assert len(set(step_ids)) == 2
