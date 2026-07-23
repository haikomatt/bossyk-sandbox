from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from bossyk_sandbox.conditions.adversary import TokenUsage
from bossyk_sandbox.instruments.base import ProposedAction
from bossyk_sandbox.instruments.policy import (
    ERROR_LABEL,
    PolicyInstrument,
    bossyk_src_path,
    default_policy_path,
)


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


class _FlakyJudge:
    """A fake matching bossyk's PolicyAwareJudge failure mode: raises like
    the real judge does when Fireworks returns malformed JSON after
    exhausting its own retries."""

    def score_step(
        self, step_id: str, action_text: str, declared_intent: str | None = None
    ) -> _FakeStepResult:
        raise RuntimeError("Judge failed after 3 attempts: Expecting property name...")


def test_policy_instrument_reports_error_label_on_judge_failure() -> None:
    instrument = PolicyInstrument(judge=_FlakyJudge())
    proposed = ProposedAction("cancel_reservation", {"reservation_id": "R1"})

    verdict = instrument.annotate(proposed, history=[])

    assert verdict.label == ERROR_LABEL
    assert "Judge failed" in verdict.detail


def test_policy_instrument_assigns_unique_step_ids_across_calls() -> None:
    judge = _FakeJudge("faithful", "ok")
    instrument = PolicyInstrument(judge=judge)

    instrument.annotate(
        ProposedAction("get_reservation_details", {"reservation_id": "R1"}), history=[]
    )
    instrument.annotate(ProposedAction("cancel_reservation", {"reservation_id": "R1"}), history=[])

    step_ids = [call[0] for call in judge.calls]
    assert len(set(step_ids)) == 2


# --- on_call token-usage hook (live-h2h4 L1 #6) ------------------------------


@dataclass
class _FakeStepResultWithUsage:
    # Mirrors bossyk's real StepResult, which carries token counts as plain
    # ints (prompt/completion) -- bossyk cannot import bossyk-sandbox's
    # TokenUsage, so the on_call contract is primitive, not a TokenUsage attr.
    label: str
    reasoning: str
    input_tokens: int
    output_tokens: int


def test_on_call_reports_usage_and_not_errored_on_success() -> None:
    class _UsageJudge:
        def score_step(
            self, step_id: str, action_text: str, declared_intent: str | None = None
        ) -> _FakeStepResultWithUsage:
            return _FakeStepResultWithUsage("faithful", "ok", input_tokens=10, output_tokens=20)

    calls: list[tuple[TokenUsage, bool]] = []
    instrument = PolicyInstrument(
        judge=_UsageJudge(), on_call=lambda usage, errored: calls.append((usage, errored))
    )

    instrument.annotate(ProposedAction("cancel_reservation", {"reservation_id": "R1"}), history=[])

    assert calls == [(TokenUsage(10, 20), False)]


def test_on_call_reports_empty_usage_when_the_judge_result_carries_none() -> None:
    judge = _FakeJudge("faithful", "ok")  # _FakeStepResult has no .usage attribute
    calls: list[tuple[TokenUsage, bool]] = []
    instrument = PolicyInstrument(
        judge=judge, on_call=lambda usage, errored: calls.append((usage, errored))
    )

    instrument.annotate(ProposedAction("cancel_reservation", {"reservation_id": "R1"}), history=[])

    assert calls == [(TokenUsage(), False)]


def test_on_call_reports_errored_true_with_empty_usage_on_judge_failure() -> None:
    calls: list[tuple[TokenUsage, bool]] = []
    instrument = PolicyInstrument(
        judge=_FlakyJudge(), on_call=lambda usage, errored: calls.append((usage, errored))
    )

    instrument.annotate(ProposedAction("cancel_reservation", {"reservation_id": "R1"}), history=[])

    assert calls == [(TokenUsage(), True)]


def test_on_call_is_never_invoked_when_unset() -> None:
    # Default behaviour (existing callers) is unchanged: no crash, nothing
    # to assert on because there's no hook.
    instrument = PolicyInstrument(judge=_FakeJudge("faithful", "ok"))

    verdict = instrument.annotate(
        ProposedAction("cancel_reservation", {"reservation_id": "R1"}), history=[]
    )

    assert verdict.label == "faithful"


# --- BOSSYK_ROOT-configurable checkout path (remediation item 2) -----------


def test_default_policy_path_honors_bossyk_root_env_var(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    custom_root = tmp_path / "custom-bossyk"
    monkeypatch.setenv("BOSSYK_ROOT", str(custom_root))

    assert default_policy_path() == custom_root / "data" / "policies" / "airline-support-v1.yaml"


def test_default_policy_path_defaults_to_projects_bossyk_under_home(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("BOSSYK_ROOT", raising=False)

    expected = (
        Path("~/Projects/bossyk").expanduser() / "data" / "policies" / ("airline-support-v1.yaml")
    )
    assert default_policy_path() == expected


def test_bossyk_src_path_honors_bossyk_root_env_var(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    custom_root = tmp_path / "custom-bossyk"
    monkeypatch.setenv("BOSSYK_ROOT", str(custom_root))

    assert bossyk_src_path() == custom_root / "src"
