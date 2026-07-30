"""RED-phase tests for outreach's Tier C control attribution and the
DETERMINISTIC-vs-JUDGED discharge distinction (bossyk-sandbox slice 2, P7).

P7 is the attestation payoff: distinguish a control discharged by a
REPRODUCIBLE CHECK (deterministic -- the evidence pack can drop the
"directional mapping" disclaimer for it) from one that is only a
structural/directional mapping (judged -- today's status quo for every
existing airline/retail Tier C tag, and the proposed default for
`ControlTag`).

Design proposed here (flagged for the RED gate, not silently decided):
- A new `ControlTag.discharge: Literal["deterministic", "judged"] = "judged"`
  field. Defaulting to "judged" means every EXISTING call site/test that
  constructs a `ControlTag` without naming `discharge` is unaffected -- the
  honest status quo (directional mapping) is preserved by default; only
  tags this phase explicitly marks pick up "deterministic".
- Outreach's Tier C entries (place_call/send_sms/send_email, book_survey,
  apply_discount) are marked "deterministic" via a NEW
  `_DETERMINISTIC_ACTION_TOOLS` frozenset in attribution.py, rather than
  widening `_ACTION_CONTROLS`'s existing 2-tuple value shape to a 3-tuple
  everywhere -- keeps every existing `_ACTION_CONTROLS` entry (cancel_
  reservation, cancel_pending_order, get_user_details, find_user_id)
  byte-identical, since none of them has a reproducible oracle backing it
  (they're structural facts about the tool, not a checked result).
- New framework/control catalogue entries this phase needs (GREEN work,
  flagged here): `pecr:reg-21` (a NEW "pecr" framework) and `fca:conc-3`
  (a new control on the EXISTING "fca" framework) -- neither exists in
  frameworks.yaml yet, so the "every ref resolves" test below fails for
  that reason today, not just a missing-symbol reason.

Out of scope this phase (per the cleanroom spec's D6 honest-null framing):
`out_of_hours_contact` (boundary 2) and `pii_cross_contact` (boundary 6)
are not yet enactable, so no Tier C mapping for them either.
"""

from __future__ import annotations

from bossyk_sandbox.compliance.attribution import ControlTag, controls_for_step
from bossyk_sandbox.compliance.frameworks import load_frameworks
from bossyk_sandbox.instruments.base import ProposedAction, Verdict

_PLACE_CALL = ProposedAction("place_call", {"phone": "+441135550001"})
_SEND_SMS = ProposedAction("send_sms", {"phone": "+441135550001", "body": "hi"})
_SEND_EMAIL = ProposedAction("send_email", {"email": "a@example.invalid", "body": "hi"})
_BOOK_SURVEY = ProposedAction("book_survey", {"prospect_id": "P-1", "slot": "2026-08-03T10:00"})
_APPLY_DISCOUNT = ProposedAction("apply_discount", {"prospect_id": "P-1", "amount": 50})


def _by_basis(tags: list[ControlTag]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for tag in tags:
        out.setdefault(tag.basis, []).append(tag.ref)
    return out


def _tag_by_ref(tags: list[ControlTag], ref: str) -> ControlTag:
    return next(tag for tag in tags if tag.ref == ref)


# --- ControlTag.discharge (new field) ---------------------------------------


def test_control_tag_discharge_defaults_to_judged() -> None:
    tag = ControlTag(ref="eu-ai-act:art-12", basis="substrate")

    assert tag.discharge == "judged"


def test_control_tag_accepts_an_explicit_deterministic_discharge() -> None:
    tag = ControlTag(
        ref="pecr:reg-21", basis="boundary:contact_without_consent", discharge="deterministic"
    )

    assert tag.discharge == "deterministic"


# --- outreach Tier C mappings ------------------------------------------------


def test_place_call_adds_the_pecr_contact_control() -> None:
    by_basis = _by_basis(controls_for_step(_PLACE_CALL, Verdict.BLOCK, overridden=False))

    assert by_basis["boundary:contact_without_consent"] == ["pecr:reg-21"]


def test_send_sms_and_send_email_add_the_same_pecr_contact_control() -> None:
    for proposed in (_SEND_SMS, _SEND_EMAIL):
        by_basis = _by_basis(controls_for_step(proposed, Verdict.ALLOW, overridden=False))
        assert by_basis["boundary:contact_without_consent"] == ["pecr:reg-21"]


def test_book_survey_adds_the_consumer_duty_control() -> None:
    by_basis = _by_basis(controls_for_step(_BOOK_SURVEY, Verdict.ALLOW, overridden=False))

    assert by_basis["boundary:booking_without_eligibility"] == ["fca:consumer-duty"]


def test_apply_discount_adds_the_consumer_duty_control() -> None:
    by_basis = _by_basis(controls_for_step(_APPLY_DISCOUNT, Verdict.BLOCK, overridden=False))

    assert by_basis["boundary:unauthorised_incentive"] == ["fca:consumer-duty"]


# --- deterministic vs judged discharge ---------------------------------------


def test_outreach_tier_c_controls_are_deterministic() -> None:
    for proposed, ref in (
        (_PLACE_CALL, "pecr:reg-21"),
        (_SEND_SMS, "pecr:reg-21"),
        (_SEND_EMAIL, "pecr:reg-21"),
        (_BOOK_SURVEY, "fca:consumer-duty"),
        (_APPLY_DISCOUNT, "fca:consumer-duty"),
    ):
        tags = controls_for_step(proposed, Verdict.ALLOW, overridden=False)
        tag = _tag_by_ref(tags, ref)
        assert tag.discharge == "deterministic", f"{proposed.tool_name} -> {ref}"


def test_airline_and_retail_tier_c_controls_stay_judged() -> None:
    # Existing structural (not oracle-backed) Tier C tags must NOT flip to
    # deterministic just because outreach's did -- re-auditing existing
    # tags' discharge status is out of scope this phase.
    cancel = ProposedAction("cancel_reservation", {"reservation_id": "R1"})
    user_lookup = ProposedAction("get_user_details", {"user_id": "U1"})

    cancel_tag = _tag_by_ref(
        controls_for_step(cancel, Verdict.BLOCK, overridden=False), "fca:consumer-duty"
    )
    lookup_tag = _tag_by_ref(
        controls_for_step(user_lookup, Verdict.ALLOW, overridden=False), "hipaa:access-logging"
    )

    assert cancel_tag.discharge == "judged"
    assert lookup_tag.discharge == "judged"


# --- catalogue resolution ----------------------------------------------------


def test_outreach_tier_c_refs_resolve_against_the_committed_catalogue() -> None:
    registry = load_frameworks()
    resolvable = {f"{f.id}:{c.id}" for f in registry.entries for c in f.controls}

    seen: set[str] = set()
    for proposed in (_PLACE_CALL, _SEND_SMS, _SEND_EMAIL, _BOOK_SURVEY, _APPLY_DISCOUNT):
        for tag in controls_for_step(proposed, Verdict.BLOCK, overridden=True):
            seen.add(tag.ref)

    assert "pecr:reg-21" in seen
    assert "fca:consumer-duty" in seen
    unresolved = seen - resolvable
    assert unresolved == set(), f"tagger emits refs not in the catalogue: {unresolved}"
