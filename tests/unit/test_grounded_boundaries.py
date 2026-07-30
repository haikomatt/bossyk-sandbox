"""RED-phase tests for the grounded-corpus grid restriction (bossyk-sandbox
slice 3 bugfix): the tool-grounded adversary's prompt-building
(`fireworks_adversary.py::_red_team_user_prompt`) calls `boundary_spec_for`
per cell whenever `tool_context` is provided (the GROUNDED path always
provides it) -- so any boundary with NO registered `BoundarySpec` crashes
generation with `KeyError`. This is what happened live: the outreach
GROUNDED grid iterated all six `OUTREACH_BOUNDARIES`, but only
contact_without_consent / booking_without_eligibility / unauthorised_incentive
have specs -- `out_of_hours_contact` (cell 2) KeyError'd, ~10 calls spent
and discarded.

Design proposed here (flagged for review, not silently decided): a
DYNAMIC, spec-derived filter -- `has_boundary_spec(boundary, domain)` and
`grounded_boundaries(domain)`, both new in `conditions/live_boundary.py` --
rather than a separately-maintained `GROUNDED_BOUNDARIES_BY_DOMAIN` list.
Rationale: a hand-maintained parallel list is EXACTLY the kind of thing that
drifts out of sync with `BOUNDARY_SPECS_BY_DOMAIN` and caused this crash in
the first place (`OUTREACH_BOUNDARIES` in conditions/grid.py already IS
such a list, and it silently went stale the moment slice 3 registered only
3 of its 6 boundaries with a spec). Deriving `grounded_boundaries` directly
from `BOUNDARY_SPECS_BY_DOMAIN`'s own keys means it can never name a
boundary the adversary-prompt code can't handle -- it is definitionally in
sync.

`has_boundary_spec` returns False (never raises) for any boundary/domain
combination `boundary_spec_for` would reject -- unregistered domain OR
unregistered boundary within a registered domain -- since its purpose is a
safe "can the grounded adversary use this boundary" check, not a strict
lookup.

`grounded_boundaries(domain)` = every boundary with ANY spec entry
(structural OR not) -- `unauthorised_incentive` (is_structural=False, an
honest null for the LIVE ORACLE) is still INCLUDED, because the adversary
can still write a payload targeting apply_discount; `boundary_spec_for`
resolving without raising is all `_red_team_user_prompt` needs. Only
boundaries with NO entry at all (out_of_hours_contact,
prohibited_financial_promotion, pii_cross_contact) are excluded.

For airline/retail, every named grid boundary already has a spec entry
(including the non-structural ones, e.g. airline's refund_over_threshold),
so `grounded_boundaries(domain) == boundaries_for(domain)` -- a NO-OP,
verified below.
"""

from __future__ import annotations

from bossyk_sandbox.conditions.grid import boundaries_for
from bossyk_sandbox.conditions.live_boundary import grounded_boundaries, has_boundary_spec


def test_has_boundary_spec_true_for_a_registered_outreach_boundary() -> None:
    assert has_boundary_spec("contact_without_consent", domain="outreach") is True
    assert has_boundary_spec("booking_without_eligibility", domain="outreach") is True
    assert has_boundary_spec("unauthorised_incentive", domain="outreach") is True


def test_has_boundary_spec_false_for_unspecced_outreach_boundaries() -> None:
    assert has_boundary_spec("out_of_hours_contact", domain="outreach") is False
    assert has_boundary_spec("prohibited_financial_promotion", domain="outreach") is False
    assert has_boundary_spec("pii_cross_contact", domain="outreach") is False


def test_has_boundary_spec_never_raises_for_an_unregistered_domain() -> None:
    assert has_boundary_spec("anything", domain="not-a-real-domain") is False


def test_grounded_boundaries_for_outreach_is_exactly_the_three_specced_boundaries() -> None:
    assert grounded_boundaries("outreach") == [
        "contact_without_consent",
        "booking_without_eligibility",
        "unauthorised_incentive",
    ]


def test_grounded_boundaries_excludes_the_utterance_boundary() -> None:
    assert "prohibited_financial_promotion" not in grounded_boundaries("outreach")


def test_grounded_boundaries_is_a_no_op_for_airline() -> None:
    # Every airline grid boundary already has a spec (including the
    # non-structural refund_over_threshold/pii_disclosure) -- the filter
    # must not remove anything.
    assert grounded_boundaries("airline") == boundaries_for("airline")


def test_grounded_boundaries_is_a_no_op_for_retail() -> None:
    assert grounded_boundaries("retail") == boundaries_for("retail")
