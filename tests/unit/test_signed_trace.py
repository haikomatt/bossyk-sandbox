"""The signed trace sidecar: the browsable evidence the pack cannot carry.
An auditk EvidencePack holds only counts (trace_summary), never the steps,
so the per-step control tags surface via a companion artifact -- the trace
plus its coverage roll-up, Ed25519-signed by the same key that signs the
pack, so the record an auditor reads is itself tamper-evident.

auditk exposes no seam to fix a pack's per-build pack_id/issued_at, but the
trace has none of those -- fixed trace/step ids and timestamps -- so the
signed trace regenerates byte-for-byte except its signature (a fresh key
each build), which is the honest determinism contract for it.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from auditk.attestation.signer import generate_keypair

from bossyk_sandbox.evidence.sample import build_sample_trace
from bossyk_sandbox.evidence.signed_trace import (
    SignedTrace,
    build_signed_trace,
    read_signed_trace,
    stable_trace_manifest,
    verify_signed_trace,
    write_signed_trace,
)

REPO_ROOT = Path(__file__).parent.parent.parent
BENCH_OUTPUT = REPO_ROOT / "docs" / "bench_output"
SAMPLE_TRACE = BENCH_OUTPUT / "evidence_pack_sample.trace.json"
SAMPLE_PUBKEY = BENCH_OUTPUT / "evidence_pack_sample.pub"


def _build(signer_key_path: Path) -> SignedTrace:
    return build_signed_trace(build_sample_trace(), signer_key_path=signer_key_path)


def test_signed_trace_carries_the_coverage_roll_up() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        priv, _ = generate_keypair(Path(tmp) / "k")
        signed = _build(priv)

    # the sample trace's steps discharge the substrate controls
    assert "eu-ai-act:art-12" in signed.coverage
    assert signed.coverage["eu-ai-act:art-12"] == [
        "sample-step-1-lookup",
        "sample-step-2-unauthorised-cancel",
    ]
    # the blocked cancel evidences the incident-response control
    assert signed.coverage["soc2:cc7-4"] == ["sample-step-2-unauthorised-cancel"]


def test_signed_trace_verifies_offline_against_its_own_key() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        priv, pub = generate_keypair(Path(tmp) / "k")
        signed = _build(priv)

        assert verify_signed_trace(signed, pub.read_text()) is True


def test_signed_trace_round_trips_through_disk(tmp_path: Path) -> None:
    priv, pub = generate_keypair(tmp_path / "k")
    signed = _build(priv)
    path = tmp_path / "sample.trace.json"

    write_signed_trace(signed, path)
    reloaded = read_signed_trace(path)

    assert verify_signed_trace(reloaded, pub.read_text()) is True
    assert reloaded.coverage == signed.coverage


def test_content_regenerates_bar_the_signature() -> None:
    # trace has no per-build volatile fields, so only the signature (fresh
    # key) differs between builds.
    with tempfile.TemporaryDirectory() as tmp:
        priv1, _ = generate_keypair(Path(tmp) / "k1")
        priv2, _ = generate_keypair(Path(tmp) / "k2")
        first = _build(priv1)
        second = _build(priv2)

    assert stable_trace_manifest(first) == stable_trace_manifest(second)


# --- the committed sample trace sidecar -------------------------------


def test_committed_sample_trace_verifies_offline() -> None:
    signed = read_signed_trace(SAMPLE_TRACE)

    assert verify_signed_trace(signed, SAMPLE_PUBKEY.read_text()) is True


def test_committed_sample_trace_content_regenerates() -> None:
    committed = read_signed_trace(SAMPLE_TRACE)

    with tempfile.TemporaryDirectory() as tmp:
        priv, _ = generate_keypair(Path(tmp) / "k")
        regenerated = _build(priv)

    assert stable_trace_manifest(regenerated) == stable_trace_manifest(committed)
