from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

from auditk.schema import Action, ActionType, Actor, DriftReport, FlowType, Step, Trace

from bossyk_sandbox.instruments.base import InstrumentVerdict, ProposedAction

UNSCORED_LABEL = "unscored"


class DriftScorer(Protocol):
    """Matches `auditk.analysis.protocols.Scorer` — satisfied by
    `auditk.analysis.scorers.get_scorer("llm-judge@0.3")` for the real
    benchmark run, or a fake for deterministic tests."""

    def score(self, trace: Trace) -> DriftReport: ...


def _step_from_action(step_id: str, proposed: ProposedAction) -> Step:
    return Step(
        step_id=step_id,
        trace_id="drift-scratch",
        timestamp=datetime.now(UTC),
        actor=Actor.AGENT,
        declared_intent=proposed.declared_intent,
        action=Action(
            type=ActionType.TOOL_CALL,
            payload={"tool_name": proposed.tool_name, "arguments": proposed.arguments},
        ),
    )


@dataclass
class DriftInstrument:
    """Wraps an auditk drift `Scorer` (`compute_drift`'s scorer registry) as
    a `SlowInstrument`. Scores intent-enactment drift for `proposed` in the
    context of `history`, without modifying auditk."""

    scorer: DriftScorer
    name: str = "drift"

    def annotate(
        self, proposed: ProposedAction, history: list[ProposedAction]
    ) -> InstrumentVerdict:
        steps = [_step_from_action(f"h{i}", action) for i, action in enumerate(history)]
        last_step_id = f"h{len(history)}"
        steps.append(_step_from_action(last_step_id, proposed))
        trace = Trace(
            trace_id="drift-scratch",
            flow_type=FlowType.GENERIC,
            agent_config_ref="drift-scratch",
            steps=steps,
            source_adapter="bossyk-sandbox-drift@0.1",
        )
        report = self.scorer.score(trace)
        step_drift = (report.per_step or {}).get(last_step_id)
        if step_drift is None:
            return InstrumentVerdict(
                instrument=self.name, label=UNSCORED_LABEL, detail="no declared_intent to score"
            )
        return InstrumentVerdict(
            instrument=self.name, label=step_drift.label.value, detail=step_drift.reasoning
        )


def build_default_drift_instrument() -> DriftInstrument:
    """Real judge path: auditk's `llm-judge@0.3` scorer (NLI gate +
    FireworksJudge/gpt-oss-120b). Requires the `[judge]` extra plus
    RUN_JUDGE_MODEL=1, RUN_NLI_MODEL=1, FIREWORKS_API_KEY — gated, not
    imported at module load time so unit tests never need torch/transformers."""
    from auditk.analysis.scorers import get_scorer

    return DriftInstrument(scorer=get_scorer("llm-judge@0.3"))
