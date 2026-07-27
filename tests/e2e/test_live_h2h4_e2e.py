from __future__ import annotations

from dataclasses import dataclass, field

from auditk.schema import ProbeDefinition

from bossyk_sandbox.conditions.live_boundary import reached_boundary
from bossyk_sandbox.conditions.live_replay import (
    LiveRunResult,
    replay_crossing,
    score_policy_post_hoc,
)
from bossyk_sandbox.conditions.retention import load_regression_probes
from bossyk_sandbox.env import REPO_ROOT
from bossyk_sandbox.instruments.base import InstrumentVerdict, ProposedAction
from bossyk_sandbox.scoring.live_h2 import by_boundary, live_h4, score_crossing

# End-to-end, fully offline: a real frozen crossing from probes/regression/
# airline.json is replayed through a FAKE live-agent session (no network, no
# model calls) that proposes the boundary-crossing tool call with no prior
# lookup, then a FAKE policy judge flags it post-hoc. Exercises the whole L1
# pipeline: replay_crossing -> the boundary oracle -> score_policy_post_hoc ->
# score_crossing -> live-H2 (group_scores/by_boundary) + live-H4 (live_h4).
# Not gated behind an env var (unlike test_walking_skeleton.py /
# test_multi_instrument_e2e.py): it's deterministic and cheap, like
# test_probe_grid_e2e.py, so it runs as part of the normal `pytest tests/`
# gate.

AIRLINE_CROSSINGS_PATH = REPO_ROOT / "probes" / "regression" / "airline.json"


@dataclass
class _ScriptedPolicy:
    """Fake SlowInstrument: fires `label` whenever `proposed` calls
    `flagged_tool`, else reports faithful -- deterministic stand-in for
    bossyk's real PolicyAwareJudge."""

    flagged_tool: str
    label: str
    name: str = "policy"
    calls: int = field(default=0)

    def annotate(
        self, proposed: ProposedAction, history: list[ProposedAction]
    ) -> InstrumentVerdict:
        self.calls += 1
        if proposed.tool_name == self.flagged_tool:
            return InstrumentVerdict(instrument=self.name, label=self.label, detail="scripted")
        return InstrumentVerdict(instrument=self.name, label="faithful", detail="scripted")


def _load_crossing_probe(family: str) -> ProbeDefinition:
    probes = load_regression_probes(AIRLINE_CROSSINGS_PATH)
    return next(p for p in probes if p.family == family)


def test_live_h2h4_pipeline_end_to_end_offline() -> None:
    probe = _load_crossing_probe("airline-cancel_without_lookup")

    # The fake live agent: proposes cancel_reservation for RES-1 with no
    # prior lookup (an unauthorised call that got through -- e.g. the gate
    # wasn't wired for this tool, or a manual override let it execute), so
    # this crossing lands in "reached, not prevented" -- the interesting
    # case for a policy judge to catch.
    culprit_call = ProposedAction("cancel_reservation", {"reservation_id": "RES-1"})

    def fake_run_session(_payload: str) -> LiveRunResult:
        return LiveRunResult(proposed=[culprit_call], executed=[culprit_call])

    replay = replay_crossing(probe, fake_run_session)

    # (a) the boundary oracle: the live session DID propose the
    # boundary-crossing action.
    assert replay.domain == "airline"
    assert replay.boundary == "cancel_without_lookup"
    assert reached_boundary(replay.boundary, replay.proposed, replay.executed, domain="airline")

    # (b) post-hoc policy scoring flags the unauthorised cancel.
    policy = _ScriptedPolicy(flagged_tool="cancel_reservation", label="instruction_noncompliance")
    scored = score_policy_post_hoc(replay, policy)
    assert policy.calls == 1
    assert scored.verdicts[0].label == "instruction_noncompliance"

    # (c) live-H2/H4 numbers come out right for this one crossing.
    score = score_crossing(replay, scored.verdicts)
    assert score.reached is True
    assert score.prevented is False
    assert score.detected is True
    assert score.caught is True
    assert score.detected_too_late is True

    summary = by_boundary([score])["cancel_without_lookup"]
    assert summary.reach.n == 1
    assert summary.reach.successes == 1
    assert summary.catch.n == 1
    assert summary.catch.successes == 1
    assert summary.n_policy_scored == 1

    h4 = live_h4([score])
    assert h4.n_violations == 1
    assert h4.prevented == 0
    assert h4.detected_too_late == 1
    assert h4.undetected == 0
    assert h4.harm_off == 1
    assert h4.harm_on == 1  # not prevented -> still executes even with the gate "on"
    assert h4.harm_delta == 0


def test_live_h2h4_pipeline_prevention_case_end_to_end_offline() -> None:
    # A second crossing, this time PREVENTED: the fake session proposes the
    # unauthorised cancel but it never executes (the gate blocked it
    # in-loop) -- caught should be True via prevention alone, with no
    # policy verdict needed at all.
    probe = _load_crossing_probe("airline-cancel_without_lookup")
    culprit_call = ProposedAction("cancel_reservation", {"reservation_id": "RES-2"})

    def fake_run_session(_payload: str) -> LiveRunResult:
        return LiveRunResult(proposed=[culprit_call], executed=[])

    replay = replay_crossing(probe, fake_run_session)
    score = score_crossing(replay)  # no policy_verdicts at all

    assert score.reached is True
    assert score.prevented is True
    assert score.caught is True
    assert score.detected_too_late is False

    h4 = live_h4([score])
    assert h4.prevented == 1
    assert h4.harm_delta == 1
