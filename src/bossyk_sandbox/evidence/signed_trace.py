"""A signed trace sidecar: the browsable evidence an auditk EvidencePack
cannot carry. The pack holds only counts (`trace_summary`), so the per-step
control tags live here -- the full trace plus its coverage roll-up,
Ed25519-signed by the same key that signs the pack, so the record an
auditor reads is itself tamper-evident.

Mirrors `evidence/pack.py`'s sign/verify path (auditk's `canonicalize` +
`LocalEd25519Signer`) rather than reinventing it. Unlike a pack, a trace
has no per-build volatile fields (fixed trace/step ids and timestamps), so
a signed trace regenerates byte-for-byte except its signature -- see
`stable_trace_manifest`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from auditk.attestation.canonical import canonicalize
from auditk.attestation.signer import LocalEd25519Signer, LocalEd25519Verifier
from auditk.schema import Signature, Trace
from pydantic import BaseModel, Field

from bossyk_sandbox.compliance.coverage import trace_control_coverage


class SignedTrace(BaseModel):
    """A trace, its control-coverage roll-up, and the signatures over both.
    Bundling the coverage into the signed payload binds it to the trace it
    summarises, so a consumer cannot be handed a roll-up that disagrees with
    the steps."""

    trace: Trace
    coverage: dict[str, list[str]]
    signatures: list[Signature] = Field(default_factory=list)


def stable_trace_manifest(signed: SignedTrace) -> dict[str, Any]:
    """The signed content without the signatures -- the byte-stable part.
    Two independently-signed bundles over the same trace compare equal here
    (only the signature, from a fresh key, differs)."""
    manifest: dict[str, Any] = signed.model_dump(mode="json", exclude={"signatures"})
    return manifest


def build_signed_trace(trace: Trace, *, signer_key_path: Path | str) -> SignedTrace:
    """Computes the coverage roll-up from the trace's tagged steps, then
    Ed25519-signs the (trace + coverage) payload. Coverage is derived here,
    never passed in, so the bundle is always self-consistent."""
    signed = SignedTrace(trace=trace, coverage=trace_control_coverage(trace))
    canonical = canonicalize(stable_trace_manifest(signed))
    signer = LocalEd25519Signer(signer_key_path)
    signed.signatures.append(signer.sign(canonical))
    return signed


def verify_signed_trace(signed: SignedTrace, public_key_pem: str) -> bool:
    """Verifies every signature against a trusted public key, following the
    same canonicalization path as the signer. False if unsigned."""
    if not signed.signatures:
        return False
    canonical = canonicalize(stable_trace_manifest(signed))
    verifier = LocalEd25519Verifier(public_key_pem)
    for signature in signed.signatures:
        try:
            verifier.verify(canonical, signature.signature)
        except Exception:
            return False
    return True


def write_signed_trace(signed: SignedTrace, path: Path | str) -> None:
    Path(path).write_text(signed.model_dump_json(indent=2))


def read_signed_trace(path: Path | str) -> SignedTrace:
    return SignedTrace.model_validate(json.loads(Path(path).read_text()))
