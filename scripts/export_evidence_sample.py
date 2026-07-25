#!/usr/bin/env python
"""Regenerates the committed sample evidence pack
`docs/bench_output/evidence_pack_sample.json` (and its verifying public
key `evidence_pack_sample.pub`) from the deterministic scripted trace in
`bossyk_sandbox.evidence.sample`.

The pack is the story's concrete proof that the observe/attest/interrupt
substrate is real: a signed EvidencePack over a two-step trace (a benign
lookup allowed, an unauthorised cancel blocked). The
`attested-action-log-is-the-substrate` claim cites it so the logging and
attestation controls (EU AI Act Art. 12, HIPAA audit/access, ISO 27001
A.8.15, ISO 42001 records, SOC 2 CC7.2) have a real artifact behind them.

Also writes the signed trace sidecar `evidence_pack_sample.trace.json` --
the steps (with their per-step compliance control tags) and the coverage
roll-up the pack itself cannot carry, signed by the same key. One committed
public key verifies both.

Deterministic content, one honest caveat: auditk stamps a fresh `pack_id`
and `issued_at` on every build with no seam to fix them, so re-running
this changes those two pack fields (and the signature over them) but
nothing else -- see `bossyk_sandbox.evidence.sample.stable_manifest`. The
trace sidecar has no such volatile fields, so it regenerates byte-for-byte
except its signature (see `evidence.signed_trace.stable_trace_manifest`). A
freshly minted signing key is generated per run and discarded; only the
public key is committed, alongside the artifacts it verifies. No network,
no secrets.

Usage:
    uv run python scripts/export_evidence_sample.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from auditk.attestation.signer import generate_keypair

from bossyk_sandbox.evidence.pack import build_and_sign, verify_offline, write_pack
from bossyk_sandbox.evidence.sample import SAMPLE_AGENT_VERSION, build_sample_trace
from bossyk_sandbox.evidence.signed_trace import (
    build_signed_trace,
    verify_signed_trace,
    write_signed_trace,
)

REPO_ROOT = Path(__file__).parent.parent
BENCH_OUTPUT = REPO_ROOT / "docs" / "bench_output"
PACK_PATH = BENCH_OUTPUT / "evidence_pack_sample.json"
TRACE_PATH = BENCH_OUTPUT / "evidence_pack_sample.trace.json"
PUBKEY_PATH = BENCH_OUTPUT / "evidence_pack_sample.pub"


def main(argv: list[str] | None = None) -> int:
    del argv  # no CLI args -- always regenerates the one committed sample
    trace = build_sample_trace()
    with tempfile.TemporaryDirectory() as tmp:
        private_key_path, public_key_path = generate_keypair(Path(tmp) / "sample-signer")
        # Same key signs the pack (counts, per-build volatile) and the trace
        # sidecar (steps + control tags), so one committed public key
        # verifies both.
        pack = build_and_sign(
            trace,
            agent_version=SAMPLE_AGENT_VERSION,
            signer_key_path=private_key_path,
        )
        signed_trace = build_signed_trace(trace, signer_key_path=private_key_path)
        public_key_pem = public_key_path.read_text()

    if not verify_offline(pack, public_key_pem):
        raise RuntimeError("sample pack failed to verify against its own freshly generated key")
    if not verify_signed_trace(signed_trace, public_key_pem):
        raise RuntimeError("sample trace failed to verify against its own freshly generated key")

    write_pack(pack, PACK_PATH)
    write_signed_trace(signed_trace, TRACE_PATH)
    PUBKEY_PATH.write_text(public_key_pem)

    print(f"export_evidence_sample: wrote {PACK_PATH.relative_to(REPO_ROOT)}")
    print(f"export_evidence_sample: wrote {TRACE_PATH.relative_to(REPO_ROOT)}")
    print(f"export_evidence_sample: wrote {PUBKEY_PATH.relative_to(REPO_ROOT)}")
    print(
        f"export_evidence_sample: {pack.trace_summary.step_count} steps, "
        f"{len(signed_trace.coverage)} controls covered, both verify offline"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
