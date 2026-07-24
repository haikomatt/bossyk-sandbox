"""The committed sample evidence pack: a small, fully scripted
observe/attest/interrupt trace that `scripts/export_evidence_sample.py`
signs and freezes to `docs/bench_output/evidence_pack_sample.json`. It
exists so the story's `attested-action-log-is-the-substrate` claim can
point at a real signed artifact rather than prose -- the logging and
attestation controls (EU AI Act Art. 12, HIPAA audit/access, ISO 27001
A.8.15, ISO 42001 records, SOC 2 CC7.2) are what the evidence pack itself
discharges, so the demo must show one.

Everything here is deterministic by construction: fixed trace/step ids and
fixed timestamps, no wall-clock read. The only non-reproducible parts of a
built pack are auditk's per-build `pack_id` and `issued_at` (it exposes no
seam for either), which `VOLATILE_PACK_FIELDS`/`stable_manifest` carve out
so a rebuild can be checked for content stability without pretending those
fields are constant.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from auditk.schema import EvidencePack, Trace

from bossyk_sandbox.evidence.trace import build_trace, make_attested_step
from bossyk_sandbox.instruments.base import Decision, ProposedAction, Verdict

SAMPLE_TRACE_ID = "bossyk-sample-attested-log"
SAMPLE_AGENT_CONFIG_REF = "retail-support-demo@sample"
SAMPLE_AGENT_VERSION = "sample-0.1"

# A fixed instant (not the wall clock) so the trace is byte-reproducible.
_SAMPLE_EPOCH = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)

# auditk's pack builder stamps these two per build (uuid4 + datetime.now)
# and offers no injectable seam; every other field is deterministic once
# the trace is.
VOLATILE_PACK_FIELDS = {"pack_id", "issued_at", "signatures"}


def build_sample_trace() -> Trace:
    """Builds the deterministic two-step demo trace: a benign order lookup
    the gate allows, then an unauthorised cancellation the gate blocks
    before execution -- the smallest audit trail that shows both an
    attested allow and an attested interrupt. No override on either step,
    so each attests its own automatic verdict."""
    lookup = make_attested_step(
        SAMPLE_TRACE_ID,
        ProposedAction(
            tool_name="get_order_details",
            arguments={"order_id": "W1001"},
            declared_intent="look up the customer's order before acting on it",
        ),
        auto_decision=Decision(Verdict.ALLOW, "read-only lookup is permitted"),
        final_verdict=Verdict.ALLOW,
        step_id="sample-step-1-lookup",
        timestamp=_SAMPLE_EPOCH,
    )
    unauthorised_cancel = make_attested_step(
        SAMPLE_TRACE_ID,
        ProposedAction(
            tool_name="cancel_pending_order",
            arguments={"order_id": "W2002"},
            declared_intent="cancel the order without a prior authorisation lookup",
        ),
        auto_decision=Decision(Verdict.BLOCK, "cancel requires a prior lookup on this order"),
        final_verdict=Verdict.BLOCK,
        step_id="sample-step-2-unauthorised-cancel",
        timestamp=_SAMPLE_EPOCH.replace(second=1),
    )
    return build_trace(SAMPLE_TRACE_ID, SAMPLE_AGENT_CONFIG_REF, [lookup, unauthorised_cancel])


def stable_manifest(pack: EvidencePack) -> dict[str, Any]:
    """The pack's content with the two unavoidably-volatile fields (and the
    signatures that sign over them) removed, so two independently-built
    packs can be compared for content equality. Deliberately narrow: adding
    a field here would let real content drift pass a regeneration check."""
    manifest: dict[str, Any] = pack.model_dump(mode="json", exclude=VOLATILE_PACK_FIELDS)
    return manifest
