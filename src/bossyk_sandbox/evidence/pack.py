from __future__ import annotations

import json
from pathlib import Path

from auditk.attestation.canonical import canonicalize
from auditk.attestation.pack import build as build_pack
from auditk.attestation.signer import LocalEd25519Signer, LocalEd25519Verifier
from auditk.schema import EvidencePack, Issuer, RiskTier, Subject, Trace


def build_and_sign(
    trace: Trace,
    *,
    agent_version: str,
    signer_key_path: Path | str,
    issuer_name: str = "bossyk-sandbox",
    risk_tier: RiskTier = RiskTier.LIMITED,
    jurisdiction: list[str] | None = None,
) -> EvidencePack:
    """Wrap auditk's pack builder + Ed25519 signer. scorer_key=None keeps this
    judge-free and deterministic (Phase 0 has no LLM in the gating path)."""
    signer = LocalEd25519Signer(signer_key_path)
    return build_pack(
        traces=[trace],
        probe_results=[],
        jurisdiction=jurisdiction or [],
        risk_tier=risk_tier,
        issuer=Issuer(name=issuer_name),
        subject=Subject(agent_config_ref=trace.agent_config_ref, agent_version=agent_version),
        signer=signer,
        scorer_key=None,
    )


def verify_offline(pack: EvidencePack, public_key_pem: str) -> bool:
    """Verify every signature on the pack against a trusted public key,
    following the same canonicalization path as `auditk verify`."""
    if not pack.signatures:
        return False
    manifest = pack.model_dump(mode="json", exclude={"signatures"})
    canonical = canonicalize(manifest)
    verifier = LocalEd25519Verifier(public_key_pem)
    for signature in pack.signatures:
        try:
            verifier.verify(canonical, signature.signature)
        except Exception:
            return False
    return True


def write_pack(pack: EvidencePack, path: Path | str) -> None:
    Path(path).write_text(pack.model_dump_json(indent=2))


def read_pack(path: Path | str) -> EvidencePack:
    return EvidencePack.model_validate(json.loads(Path(path).read_text()))
