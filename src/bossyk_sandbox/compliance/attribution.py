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

from bossyk_sandbox.instruments.base import Verdict

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


class ControlTag(BaseModel):
    """One compliance control an attested action discharges, with the
    `basis` for why it applies (substrate, verdict:gated, verdict:blocked,
    verdict:overridden). `ref` is a `<framework-id>:<control-id>` that
    resolves against the shared catalogue."""

    ref: str
    basis: str


def controls_for_step(verdict: Verdict, *, overridden: bool) -> list[ControlTag]:
    """The compliance controls a gated, attested step discharges, from what
    is known at attestation time: that it is a signed log entry (substrate)
    and its final verdict / override (verdict-derived). Deterministic order
    -- substrate, then gated, then blocked, then overridden -- and free of
    duplicate refs. Nothing here inspects the tool call; action-specific
    (boundary / PII) tags are a later phase."""
    tags = [ControlTag(ref=ref, basis="substrate") for ref in _SUBSTRATE_CONTROLS]
    tags += [ControlTag(ref=ref, basis="verdict:gated") for ref in _GATED_CONTROLS]
    if verdict is Verdict.BLOCK:
        tags += [ControlTag(ref=ref, basis="verdict:blocked") for ref in _BLOCKED_CONTROLS]
    if overridden:
        tags += [ControlTag(ref=ref, basis="verdict:overridden") for ref in _OVERRIDDEN_CONTROLS]
    return tags
