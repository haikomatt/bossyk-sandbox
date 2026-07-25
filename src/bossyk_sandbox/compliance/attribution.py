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

from pydantic import BaseModel

from bossyk_sandbox.instruments.base import ProposedAction, Verdict

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
}


class ControlTag(BaseModel):
    """One compliance control an attested action discharges, with the
    `basis` for why it applies (substrate, verdict:gated, verdict:blocked,
    verdict:overridden). `ref` is a `<framework-id>:<control-id>` that
    resolves against the shared catalogue."""

    ref: str
    basis: str


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
        tags += [ControlTag(ref=ref, basis=basis) for ref in refs]

    seen: set[str] = set()
    deduped: list[ControlTag] = []
    for tag in tags:
        if tag.ref in seen:
            continue
        seen.add(tag.ref)
        deduped.append(tag)
    return deduped
