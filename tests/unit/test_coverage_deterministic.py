"""RED-phase tests for coverage reporting's deterministic-vs-judged
distinction (bossyk-sandbox slice 2, P7): the evidence pack needs to report
which discharged controls are backed by a reproducible check (so the
"directional mapping" disclaimer can be dropped for that subset) versus
which are still a structural/judged mapping.

Proposed here: a NEW function, `trace_deterministic_control_refs`, additive
alongside the existing `trace_control_coverage` (left completely untouched,
so its existing tests in test_coverage.py stay green) -- reads the same
per-step `CONTROLS_METADATA_KEY` metadata `make_attested_step` already
writes (each serialized `ControlTag` dict picks up a `discharge` key for
free once `ControlTag.discharge` exists, since `make_attested_step` calls
`tag.model_dump()`), filtering to refs with at least one
deterministic-basis tag anywhere in the trace.
"""

from __future__ import annotations

from auditk.schema import Trace

from bossyk_sandbox.compliance.coverage import trace_deterministic_control_refs
from bossyk_sandbox.evidence.trace import build_trace, make_attested_step, make_step
from bossyk_sandbox.instruments.base import Decision, ProposedAction, Verdict


def _trace_with_outreach_and_neutral_steps() -> Trace:
    deterministic_step = make_attested_step(
        "t-1",
        ProposedAction("place_call", {"phone": "+441135550001"}),
        auto_decision=Decision(Verdict.ALLOW, "clean check"),
        final_verdict=Verdict.ALLOW,
        step_id="step-place-call",
    )
    judged_step = make_attested_step(
        "t-1",
        ProposedAction("cancel_reservation", {"reservation_id": "R1"}),
        auto_decision=Decision(Verdict.BLOCK, "no lookup"),
        final_verdict=Verdict.BLOCK,
        step_id="step-cancel",
    )
    return build_trace("t-1", "cfg", [deterministic_step, judged_step])


def test_trace_deterministic_control_refs_returns_a_set() -> None:
    refs = trace_deterministic_control_refs(_trace_with_outreach_and_neutral_steps())

    assert isinstance(refs, set)


def test_deterministic_control_refs_include_the_outreach_pecr_control() -> None:
    refs = trace_deterministic_control_refs(_trace_with_outreach_and_neutral_steps())

    assert "pecr:reg-21" in refs


def test_deterministic_control_refs_exclude_judged_only_controls() -> None:
    refs = trace_deterministic_control_refs(_trace_with_outreach_and_neutral_steps())

    # fca:consumer-duty here comes ONLY from the judged cancel_reservation
    # tag (no outreach step in this trace discharges it) -- must not appear.
    assert "fca:consumer-duty" not in refs
    # substrate/gated controls are judged (Tier A/B untouched this phase).
    assert "eu-ai-act:art-12" not in refs


def test_deterministic_control_refs_of_an_untagged_trace_is_empty() -> None:
    step = make_step("t-1", ProposedAction("x", {}), Decision(Verdict.ALLOW, "r"), step_id="s")

    refs = trace_deterministic_control_refs(build_trace("t-1", "cfg", [step]))

    assert refs == set()
