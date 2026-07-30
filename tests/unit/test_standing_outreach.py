"""RED-phase tests for `outreach_standing_grants()` (bossyk-sandbox slice 1).

Spec ("Standing grants (outreach)"):
    unauthorised_incentive   -> StandingGrant(max_amount=<caller discount ceiling>)
    booking_without_eligibility -> gated by the passed-check rule, not standing count

Only the amount-gated incentive grant is asserted here (D5/D7's slice-1
scope: the two PRECEDENCE-shaped boundaries). The full per-domain
`BOUNDARY_BY_TOOL_BY_DOMAIN` promotion D5 describes is left open for the RED
gate discussion (see report) rather than pinned by a test here.
"""

from __future__ import annotations

from bossyk_sandbox.standing import outreach_standing_grants


def test_outreach_standing_grants_gates_the_incentive_boundary_by_amount() -> None:
    grants = outreach_standing_grants()

    assert "unauthorised_incentive" in grants
    grant = grants["unauthorised_incentive"]
    assert grant.max_amount is not None
    assert grant.max_amount > 0


def test_outreach_standing_grants_does_not_gate_booking_by_standing() -> None:
    # booking_without_eligibility is gated by RequireLookupBeforeCancel
    # (outreach_fast_rules), not by a standing count/amount grant.
    grants = outreach_standing_grants()

    assert "booking_without_eligibility" not in grants
