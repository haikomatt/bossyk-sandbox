from __future__ import annotations

from auditk.analysis.taxonomy import TaxonomyLabel
from auditk.schema import DriftReport, StepDrift, Trace

from bossyk_sandbox.instruments.base import ProposedAction
from bossyk_sandbox.instruments.drift import ERROR_LABEL, DriftInstrument


class _FakeScorer:
    """A fake `auditk.analysis.protocols.Scorer` — mirrors the real
    llm-judge@0.3 skip-if-no-declared_intent behaviour so tests stay
    deterministic without torch/transformers/Fireworks."""

    method = "fake"
    method_version = "0.0"

    def __init__(self, label: TaxonomyLabel, reasoning: str) -> None:
        self._label = label
        self._reasoning = reasoning

    def score(self, trace: Trace) -> DriftReport:
        last_step = trace.steps[-1]
        if last_step.declared_intent is None:
            return DriftReport(
                drift_score=0.0, method=self.method, method_version=self.method_version
            )
        return DriftReport(
            drift_score=0.0,
            method=self.method,
            method_version=self.method_version,
            per_step={
                last_step.step_id: StepDrift(
                    step_id=last_step.step_id, label=self._label, reasoning=self._reasoning
                )
            },
        )


def test_drift_instrument_reports_faithful_label() -> None:
    instrument = DriftInstrument(
        scorer=_FakeScorer(TaxonomyLabel.FAITHFUL, "matches declared intent")
    )
    proposed = ProposedAction(
        "get_reservation_details",
        {"reservation_id": "R1"},
        declared_intent="look up the reservation before cancelling",
    )

    verdict = instrument.annotate(proposed, history=[])

    assert verdict.instrument == "drift"
    assert verdict.label == "faithful"


def test_drift_instrument_reports_flagged_label_with_history() -> None:
    instrument = DriftInstrument(
        scorer=_FakeScorer(TaxonomyLabel.GOAL_DEVIATION, "action contradicts declared intent")
    )
    history = [
        ProposedAction(
            "get_reservation_details",
            {"reservation_id": "R1"},
            declared_intent="look up the reservation",
        )
    ]
    proposed = ProposedAction(
        "cancel_reservation", {"reservation_id": "R1"}, declared_intent="check flight status"
    )

    verdict = instrument.annotate(proposed, history=history)

    assert verdict.label == "goal_deviation"
    assert "contradicts" in verdict.detail


class _RaisingScorer:
    method = "fake"
    method_version = "0.0"

    def score(self, trace: Trace) -> DriftReport:
        raise RuntimeError("judge failed after 3 attempts")


def test_drift_instrument_reports_error_label_on_scorer_failure() -> None:
    instrument = DriftInstrument(scorer=_RaisingScorer())
    proposed = ProposedAction(
        "cancel_reservation", {"reservation_id": "R1"}, declared_intent="cancel it"
    )

    verdict = instrument.annotate(proposed, history=[])

    assert verdict.label == ERROR_LABEL
    assert "judge failed" in verdict.detail


def test_drift_instrument_unscored_without_declared_intent() -> None:
    instrument = DriftInstrument(scorer=_FakeScorer(TaxonomyLabel.FAITHFUL, "n/a"))
    proposed = ProposedAction("get_user_details", {"user_id": "U1"})

    verdict = instrument.annotate(proposed, history=[])

    assert verdict.label == "unscored"


class _CapturingScorer:
    """Records the `Trace` it was called with so a test can inspect exactly
    what `DriftInstrument` hands to auditk's scorer -- in particular the
    `action.payload["text"]` field auditk's `_action_text()` requires (see
    docs/drift-diagnostic-findings.md: a missing "text" key makes auditk fall
    back to `str(dict)`, which collapses the NLI gate to `neutral`
    universally)."""

    method = "fake"
    method_version = "0.0"

    def __init__(self) -> None:
        self.last_trace: Trace | None = None

    def score(self, trace: Trace) -> DriftReport:
        self.last_trace = trace
        last_step = trace.steps[-1]
        return DriftReport(
            drift_score=0.0,
            method=self.method,
            method_version=self.method_version,
            per_step={
                last_step.step_id: StepDrift(
                    step_id=last_step.step_id, label=TaxonomyLabel.FAITHFUL, reasoning="n/a"
                )
            },
        )


def test_drift_instrument_renders_action_payload_as_natural_language_text() -> None:
    scorer = _CapturingScorer()
    instrument = DriftInstrument(scorer=scorer)
    proposed = ProposedAction(
        "cancel_reservation",
        {"reservation_id": "R1"},
        declared_intent="cancel it",
    )

    instrument.annotate(proposed, history=[])

    assert scorer.last_trace is not None
    text = scorer.last_trace.steps[-1].action.payload.get("text")
    assert isinstance(text, str)
    assert "{'" not in text  # not a Python dict repr
    assert "cancel_reservation" in text
    assert "R1" in text
