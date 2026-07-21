from __future__ import annotations

from auditk.analysis.taxonomy import TaxonomyLabel
from auditk.schema import DriftReport, StepDrift, Trace

from bossyk_sandbox.instruments.base import ProposedAction
from bossyk_sandbox.instruments.drift import DriftInstrument


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


def test_drift_instrument_unscored_without_declared_intent() -> None:
    instrument = DriftInstrument(scorer=_FakeScorer(TaxonomyLabel.FAITHFUL, "n/a"))
    proposed = ProposedAction("get_user_details", {"user_id": "U1"})

    verdict = instrument.annotate(proposed, history=[])

    assert verdict.label == "unscored"
