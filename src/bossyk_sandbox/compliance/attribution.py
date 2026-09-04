"""The evidence-pack control tagger: which compliance controls a single
attested action discharges. It answers, per governed action, "this is the
record for Art. 12 (automatic event logging) and SOC 2 CC7.3 (security-event
evaluation)," so the signed evidence speaks the auditor's controls, not just
a slide.

Tiered by what is knowable at attestation time (the phase order of the
feature):

  - **substrate** -- every attested step, because it is an observed, gated,
    signed log entry. No new signal.
  - **verdict-derived** -- from the gate verdict and whether a human
    overrode it. Already on the step.
  - action-specific (boundary / PII) tags need a signal that does not exist
    at attestation time yet and are a later phase; nothing here reads the
    tool name.

Each tag carries a `basis` naming why it applies -- the honesty guardrail
carried down from the story coverage work, made per-action, so a thin
attribution can never masquerade as a strong one. Every ref emitted here
resolves against the shared catalogue (`compliance/frameworks.yaml`); a test
enforces it, so this module stays pure (no catalogue load at tag time).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from bossyk_sandbox.instruments.base import Decision, ProposedAction, Verdict

# Namespaced like `bossyk_sandbox_verdict` (scenarios/runner.py): the tags
# ride in the auditk Step's free-form metadata dict, no schema change.
CONTROLS_METADATA_KEY = "bossyk_sandbox_controls"

# Substrate: discharged by every attested step, because the step itself is a
# signed, gated event-log entry.
_SUBSTRATE_CONTROLS = (
    "eu-ai-act:art-12",  # automatic event logging
    "iso-27001:a-8-15",  # logging
    "iso-42001:records",  # documented information
    "soc2:cc7-2",  # monitoring
    "hipaa:audit-controls",  # audit controls
)
# Gated: every attested action passed through the gate's risk control.
_GATED_CONTROLS = (
    "eu-ai-act:art-9",  # risk management system
    "soc2:cc7-3",  # security-event evaluation
)
# Blocked: the gate stopped the action -- the block IS the remediation.
_BLOCKED_CONTROLS = (
    "soc2:cc7-4",  # incident response / remediation
    "fca:sysc",  # systems and controls
)
# Overridden: a human made the final call on the automatic verdict.
_OVERRIDDEN_CONTROLS = ("eu-ai-act:art-14",)  # human oversight

# Tier C -- action-specific controls, keyed by the tool's structural purpose,
# NOT a content heuristic on its arguments. Each entry is a defensible fact
# about what the tool is: a cancellation tool is a consumer-facing mutation;
# a user-lookup tool accesses personal data by design. Content-based PII
# detection (inspecting arguments or results) would be fuzzy and
# over-claim-prone, so it is deliberately NOT done here -- see the phase doc.
# `basis` names the reason (boundary:<kind> / data-class:<kind>) so a reader
# sees exactly why the control applies.
_ACTION_CONTROLS: dict[str, tuple[str, tuple[str, ...]]] = {
    # cancellation tools -- an unauthorised or mistaken cancellation is the
    # consumer harm FCA Consumer Duty is about.
    "cancel_reservation": ("boundary:cancellation", ("fca:consumer-duty",)),
    "cancel_pending_order": ("boundary:cancellation", ("fca:consumer-duty",)),
    # user-lookup tools access personal data by design, so any call is a
    # PHI/PII access event.
    "get_user_details": (
        "data-class:personal-data",
        ("hipaa:access-logging", "hipaa:minimum-necessary", "iso-27001:a-5-15"),
    ),
    "find_user_id": (
        "data-class:personal-data",
        ("hipaa:access-logging", "iso-27001:a-5-15"),
    ),
    # Outreach (bossyk-sandbox slice 2, P7) -- boundary 1
    # (contact_without_consent): PECR reg 21 governs live marketing calls
    # to a number that hasn't consented / is TPS-registered.
    "place_call": ("boundary:contact_without_consent", ("pecr:reg-21",)),
    "send_sms": ("boundary:contact_without_consent", ("pecr:reg-21",)),
    "send_email": ("boundary:contact_without_consent", ("pecr:reg-21",)),
    # Outreach boundary 3 (booking_without_eligibility) and boundary 4
    # (unauthorised_incentive): both are Consumer Duty consumer-harm acts
    # (booking an ineligible prospect / exceeding discount authority),
    # mirroring the existing cancellation->consumer-duty mapping above.
    "book_survey": ("boundary:booking_without_eligibility", ("fca:consumer-duty",)),
    "apply_discount": ("boundary:unauthorised_incentive", ("fca:consumer-duty",)),
}

# Tool names whose Tier-C control is discharged by a REPRODUCIBLE CHECK
# (deterministic -- the evidence pack can drop the "directional mapping"
# disclaimer for it), not merely a structural/directional inference
# (judged, ControlTag's default). Membership REQUIRES a wired gate whose
# result is recorded in the trace, enforced by
# `tests/unit/test_discharge_invariant.py`:
#   - place_call  <- RequirePassedCheck on check_suppression (boundary 1)
#   - book_survey <- RequirePassedCheck on check_eligibility (boundary 3)
# Both are wired in `scenarios.runner.outreach_fast_rules`, so their
# discharge is reproducible offline from the signed trace.
#
# NOT members, and why (narrowed 2026-09-04, see below):
#   - send_sms / send_email: share boundary 1's policy line but have NO rule
#     wired to them in any domain, so there is no check result in the trace
#     to reproduce. Adding a RequirePassedCheck on check_suppression for both
#     would earn them a place here; until then the claim is false.
#   - apply_discount: its justification was self-referential (the amount is
#     in the tool's own arguments), which is not an independent oracle. The
#     standing/amount check that does exist (`standing.evaluate_authority`)
#     is reachable only from the console entry points, never from
#     `runtime.langgraph_agent`, which is what produces scenario evidence.
#
# Airline/retail's existing Tier-C tags (a cancellation tool "is" a
# consumer-facing mutation; a user-lookup tool "is" a personal-data access)
# have no oracle behind them either -- they stay judged, unchanged, by not
# appearing here.
_DETERMINISTIC_ACTION_TOOLS: frozenset[str] = frozenset({"place_call", "book_survey"})

# Boundary 5 (prohibited_financial_promotion): the control a BLOCKED
# utterance discharges. The closed regulated-phrase-list match
# (instruments.utterance_rule.ProhibitedPhraseRule) is reproducible offline
# from the recorded text, so this is deterministic -- unlike every Tier C
# tool-call tag above, there is no ProposedAction here at all (see
# `controls_for_utterance`).
_UTTERANCE_BLOCKED_CONTROLS: tuple[str, ...] = ("fca:conc-3",)


class ControlTag(BaseModel):
    """One compliance control an attested action discharges, with the
    `basis` for why it applies (substrate, verdict:gated, verdict:blocked,
    verdict:overridden). `ref` is a `<framework-id>:<control-id>` that
    resolves against the shared catalogue.

    `discharge` (bossyk-sandbox slice 2, P7) distinguishes a control backed
    by a REPRODUCIBLE CHECK ("deterministic" -- the evidence pack can drop
    the "directional mapping" disclaimer for it) from one that is only a
    structural/directional inference ("judged"). Defaults to "judged" --
    the honest status quo before this phase -- so every existing call site
    that builds a `ControlTag` without naming `discharge` is unaffected."""

    ref: str
    basis: str
    discharge: Literal["deterministic", "judged"] = "judged"


def controls_for_step(
    proposed: ProposedAction, verdict: Verdict, *, overridden: bool
) -> list[ControlTag]:
    """The compliance controls a gated, attested step discharges: the
    record-keeping substrate (always), the verdict-derived controls (from
    the final verdict / override), and any action-specific controls the
    tool's structural purpose implies (Tier C). Deterministic order --
    substrate, gated, blocked, overridden, then action-specific -- and free
    of duplicate refs (an action-specific ref already present in an earlier
    tier keeps its first, stronger basis)."""
    tags = [ControlTag(ref=ref, basis="substrate") for ref in _SUBSTRATE_CONTROLS]
    tags += [ControlTag(ref=ref, basis="verdict:gated") for ref in _GATED_CONTROLS]
    if verdict is Verdict.BLOCK:
        tags += [ControlTag(ref=ref, basis="verdict:blocked") for ref in _BLOCKED_CONTROLS]
    if overridden:
        tags += [ControlTag(ref=ref, basis="verdict:overridden") for ref in _OVERRIDDEN_CONTROLS]

    action = _ACTION_CONTROLS.get(proposed.tool_name)
    if action is not None:
        basis, refs = action
        discharge: Literal["deterministic", "judged"] = (
            "deterministic" if proposed.tool_name in _DETERMINISTIC_ACTION_TOOLS else "judged"
        )
        tags += [ControlTag(ref=ref, basis=basis, discharge=discharge) for ref in refs]

    seen: set[str] = set()
    deduped: list[ControlTag] = []
    for tag in tags:
        if tag.ref in seen:
            continue
        seen.add(tag.ref)
        deduped.append(tag)
    return deduped


def controls_for_utterance(decision: Decision) -> list[ControlTag]:
    """The compliance controls a scored UTTERANCE discharges (bossyk-sandbox
    slice 2, P7; boundary 5, prohibited_financial_promotion). Unlike
    `controls_for_step`, there is no `ProposedAction` here -- P6's
    `agent_node` scores free text against `UtteranceInstrument.score`, with
    no tool call involved at all -- so this is a separate, narrower entry
    point that needs only the `Decision` the utterance rule produced.

    Only fires on BLOCK, mirroring `controls_for_step`'s `verdict:blocked`
    tier (the block IS the compliance-relevant event): an ALLOWed utterance
    (no regulated phrase found) discharges nothing. The regulated-phrase
    match is reproducible offline from the recorded text, so the control is
    tagged `discharge="deterministic"`."""
    if decision.verdict is not Verdict.BLOCK:
        return []
    return [
        ControlTag(ref=ref, basis="utterance:blocked", discharge="deterministic")
        for ref in _UTTERANCE_BLOCKED_CONTROLS
    ]
