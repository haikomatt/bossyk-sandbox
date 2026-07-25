"""The pack-level control coverage roll-up: from the per-step control tags
(written by `make_attested_step`), which controls the whole trace
discharges and which steps evidence each one. This is what an auditor reads
as "this evidence pack is the record for these controls, here are the
actions behind each."
"""

from __future__ import annotations

from auditk.schema import Trace

from bossyk_sandbox.compliance.coverage import trace_control_coverage
from bossyk_sandbox.evidence.trace import build_trace, make_attested_step
from bossyk_sandbox.instruments.base import Decision, ProposedAction, Verdict


def _sample_trace() -> Trace:
    allow = make_attested_step(
        "t-1",
        ProposedAction("get_order", {"order_id": "W1"}),
        auto_decision=Decision(Verdict.ALLOW, "lookup ok"),
        final_verdict=Verdict.ALLOW,
        step_id="step-allow",
    )
    blocked_override = make_attested_step(
        "t-1",
        ProposedAction("cancel_order", {"order_id": "W2"}),
        auto_decision=Decision(Verdict.ALLOW, "auto allowed"),
        final_verdict=Verdict.BLOCK,  # human blocked -> blocked AND overridden
        step_id="step-block",
    )
    return build_trace("t-1", "cfg", [allow, blocked_override])


def test_substrate_controls_are_evidenced_by_every_step() -> None:
    coverage = trace_control_coverage(_sample_trace())

    assert coverage["eu-ai-act:art-12"] == ["step-allow", "step-block"]
    assert coverage["soc2:cc7-2"] == ["step-allow", "step-block"]


def test_blocked_and_overridden_controls_point_only_at_the_blocked_step() -> None:
    coverage = trace_control_coverage(_sample_trace())

    assert coverage["soc2:cc7-4"] == ["step-block"]  # blocked
    assert coverage["eu-ai-act:art-14"] == ["step-block"]  # overridden


def test_coverage_keys_are_only_controls_some_step_discharges() -> None:
    coverage = trace_control_coverage(_sample_trace())

    # the allowed-only step never triggers blocked/overridden, but both steps
    # are gated, so gated controls appear
    assert "eu-ai-act:art-9" in coverage
    # nothing emits fca:consumer-duty here (that is a Tier-C / other basis)
    assert "fca:consumer-duty" not in coverage


def test_coverage_of_an_untagged_trace_is_empty() -> None:
    from bossyk_sandbox.evidence.trace import make_step

    step = make_step("t-1", ProposedAction("x", {}), Decision(Verdict.ALLOW, "r"), step_id="s")
    coverage = trace_control_coverage(build_trace("t-1", "cfg", [step]))

    assert coverage == {}
