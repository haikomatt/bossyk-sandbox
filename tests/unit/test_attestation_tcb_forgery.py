"""The attestation-TCB forgery demonstration.

A signature proves a signed trace has not been *altered since signing*. It
proves nothing about the *fidelity of capture* -- whether the tracer that
built the trace told the truth about the run. The tracer
(`build_trace`/`make_attested_step`) and the process holding the local
signing key are inside the trusted computing base (TCB); everything they
emit is trusted by construction. This test makes that boundary concrete and
undeniable.

Two arms, both ending in ``verify_signed_trace(...) is True`` against the
*same* trusted public key:

* the **honest** arm signs a real gated trace whose blocked violation is
  faithfully recorded;
* the **compromised-tracer** arm stands in for a doctored ``build_trace``
  that, from the same dirty run, emits a trace which either OMITS the
  blocked violation or RELABELS it as allowed, then signs with the same
  local key.

A verifier holding only the public key and a signed bundle cannot tell the
forged record from the honest one: both verify, yet the forged bundle
misrepresents what happened. `forge_trace` is a TEST-ONLY adversary -- it
deliberately lives here, not in the shippable package, so no forgery tool
ships. It exists to prove a property of signatures, not to add a capability.

No network, no spend: deterministic, using the fresh-keypair pattern from
`scripts/export_evidence_sample.py`.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from auditk.attestation.signer import generate_keypair
from auditk.schema import Trace

from bossyk_sandbox.compliance.attribution import CONTROLS_METADATA_KEY
from bossyk_sandbox.evidence.sample import build_sample_trace
from bossyk_sandbox.evidence.signed_trace import (
    SignedTrace,
    build_signed_trace,
    stable_trace_manifest,
    verify_signed_trace,
)
from bossyk_sandbox.evidence.trace import build_trace, make_attested_step
from bossyk_sandbox.instruments.base import Decision, ProposedAction, Verdict

# The blocked step in the sample trace and the incident-response control that
# only a *blocked* violation discharges. Its presence in a trace's coverage
# roll-up is the fingerprint of an honestly-recorded interrupt.
BLOCKED_STEP_ID = "sample-step-2-unauthorised-cancel"
BLOCK_ONLY_CONTROL = "soc2:cc7-4"


def _blocked_step_ids(trace: Trace) -> list[str]:
    return [s.step_id for s in trace.steps if s.metadata.get("gate_verdict") == "block"]


def forge_trace(trace: Trace, *, mode: str) -> Trace:
    """A COMPROMISED tracer standing in for a doctored ``build_trace``: given
    the honestly-built trace of a dirty run, it returns a trace that lies
    about that run while remaining internally self-consistent (so the
    coverage roll-up derived from it, and the signature over the pair, betray
    nothing).

    ``mode``:
      * ``"omit"``   -- drop every blocked step, so the record shows a clean
        run in which no violation ever occurred.
      * ``"relabel"`` -- rewrite each blocked step as an allowed one, re-tagged
        via the *legitimate* tracer machinery, so the violation reads as
        permitted and carries no block-only control tags.

    This is a test adversary, not product code.
    """
    if mode == "omit":
        kept = [s for s in trace.steps if s.metadata.get("gate_verdict") != "block"]
        return build_trace(trace.trace_id, trace.agent_config_ref, kept, trace.source_adapter)

    if mode == "relabel":
        steps = []
        for step in trace.steps:
            if step.metadata.get("gate_verdict") != "block":
                steps.append(step)
                continue
            # Feed the *legitimate* tracer machinery a lie: the same proposed
            # action, re-attested as an ALLOW. It comes back tagged exactly
            # like a genuinely permitted step -- no block-only controls.
            proposed = ProposedAction(
                tool_name=step.action.payload["tool_name"],
                arguments=step.action.payload["arguments"],
                declared_intent=step.declared_intent,
            )
            steps.append(
                make_attested_step(
                    trace.trace_id,
                    proposed,
                    auto_decision=Decision(Verdict.ALLOW, "action permitted"),
                    final_verdict=Verdict.ALLOW,
                    step_id=step.step_id,
                    timestamp=step.timestamp,
                )
            )
        return build_trace(trace.trace_id, trace.agent_config_ref, steps, trace.source_adapter)

    raise ValueError(f"unknown forgery mode: {mode!r}")


def _sign(trace: Trace, key_path: Path) -> SignedTrace:
    return build_signed_trace(trace, signer_key_path=key_path)


# --- the honest baseline ------------------------------------------------


def test_honest_trace_faithfully_records_the_blocked_violation() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        priv, pub = generate_keypair(Path(tmp) / "signer")
        signed = _sign(build_sample_trace(), priv)

        assert verify_signed_trace(signed, pub.read_text()) is True
        # the blocked cancel is present and its incident-response control is
        # discharged -- the record tells the truth.
        assert BLOCKED_STEP_ID in [s.step_id for s in signed.trace.steps]
        assert signed.coverage[BLOCK_ONLY_CONTROL] == [BLOCKED_STEP_ID]


# --- the forgery: a clean, correctly-signed trace of a dirty run --------


def test_omission_forgery_verifies_yet_erases_the_violation() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        # the SAME local key signs both -- a compromised tracer holds it.
        priv, pub = generate_keypair(Path(tmp) / "signer")
        pubkey = pub.read_text()

        honest = _sign(build_sample_trace(), priv)
        forged = _sign(forge_trace(build_sample_trace(), mode="omit"), priv)

        # both verify against the one trusted public key.
        assert verify_signed_trace(honest, pubkey) is True
        assert verify_signed_trace(forged, pubkey) is True

        # yet the forged record has erased the violation: no blocked step,
        # and the incident-response control it should have discharged is
        # simply absent from a fully self-consistent, signed coverage roll-up.
        assert _blocked_step_ids(forged.trace) == []
        assert BLOCK_ONLY_CONTROL not in forged.coverage
        assert BLOCK_ONLY_CONTROL in honest.coverage


def test_relabel_forgery_verifies_yet_whitewashes_the_violation() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        priv, pub = generate_keypair(Path(tmp) / "signer")
        pubkey = pub.read_text()

        forged = _sign(forge_trace(build_sample_trace(), mode="relabel"), priv)

        assert verify_signed_trace(forged, pubkey) is True

        # the cancel is still in the record -- but now it reads as ALLOWED,
        # carrying none of the block-only control tags, indistinguishable
        # from a genuinely permitted action.
        cancel = next(s for s in forged.trace.steps if s.step_id == BLOCKED_STEP_ID)
        assert cancel.metadata["gate_verdict"] == "allow"
        refs = [tag["ref"] for tag in cancel.metadata.get(CONTROLS_METADATA_KEY, [])]
        assert BLOCK_ONLY_CONTROL not in refs
        assert _blocked_step_ids(forged.trace) == []


def test_verifier_cannot_distinguish_forgery_from_truth() -> None:
    # The whole point, stated as one assertion: holding only the public key,
    # a verifier accepts the honest and the forged bundle alike, though their
    # signed contents differ -- the signature cannot speak to capture fidelity.
    with tempfile.TemporaryDirectory() as tmp:
        priv, pub = generate_keypair(Path(tmp) / "signer")
        pubkey = pub.read_text()

        honest = _sign(build_sample_trace(), priv)
        omitted = _sign(forge_trace(build_sample_trace(), mode="omit"), priv)
        relabelled = _sign(forge_trace(build_sample_trace(), mode="relabel"), priv)

        assert all(verify_signed_trace(b, pubkey) is True for b in (honest, omitted, relabelled))
        # they are genuinely different records -- the forgeries are not the
        # honest trace, they only verify like it.
        assert stable_trace_manifest(omitted) != stable_trace_manifest(honest)
        assert stable_trace_manifest(relabelled) != stable_trace_manifest(honest)
