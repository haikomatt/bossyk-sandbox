"""RED-phase tests for `ObservedAction` (bossyk-sandbox slice 2, P5).

D3 (phase-outreach-domain.md, RESOLVED): a frozen dataclass mirroring the
EXISTING `standing.TimedAction` precedent, which already wraps a
`ProposedAction` with extra data (there: `at`/`amount`; here: `result`).
`ProposedAction` itself is NOT extended with a `result` field -- a proposal
has no result, only an observed action does.

Module target chosen for this RED pass (flagged for review, not silently
decided): `ObservedAction` + its `_action_of` unwrapper live in
`instruments/base.py`, alongside `ProposedAction`/`Decision` -- both
`RequireLookupBeforeCancel` (existing, needs to keep working per the
compatibility requirement) and the new `RequirePassedCheck` (in
`instruments/hardcoded_rule.py`) need the same unwrapper, so it is defined
once and imported by both rather than duplicated.
"""

from __future__ import annotations

import pytest

from bossyk_sandbox.instruments.base import ObservedAction, ProposedAction, _action_of


def test_observed_action_wraps_a_proposed_action_and_a_result() -> None:
    action = ProposedAction("check_suppression", {"phone": "+441135550007"})

    observed = ObservedAction(action=action, result={"on_tps": True, "opted_out": False})

    assert observed.action is action
    assert observed.result == {"on_tps": True, "opted_out": False}


def test_observed_action_is_frozen() -> None:
    action = ProposedAction("check_suppression", {"phone": "+44"})
    observed = ObservedAction(action=action, result=None)

    with pytest.raises(AttributeError):
        observed.result = "mutated"  # type: ignore[misc]


def test_action_of_unwraps_a_bare_proposed_action_to_itself() -> None:
    action = ProposedAction("get_order_details", {"order_id": "#W1"})

    assert _action_of(action) is action


def test_action_of_unwraps_an_observed_action_to_its_underlying_action() -> None:
    action = ProposedAction("check_eligibility", {"prospect_id": "P-1"})
    observed = ObservedAction(action=action, result={"eligible": True})

    assert _action_of(observed) is action
