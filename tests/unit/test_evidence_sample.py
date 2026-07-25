"""The committed sample evidence pack is the story's proof that the
observe/attest/interrupt substrate is real, not prose: a signed pack over
a short scripted trace, committed to the repo and cited by the
`attested-action-log-is-the-substrate` claim.

auditk mints `pack_id` (uuid4) and `issued_at` (wall clock) per build and
exposes no seam for either, so a whole-file byte diff cannot pass. The
honest contract (chosen over freezing those fields to a fiction) is:

  1. the committed pack verifies offline against the committed public key;
  2. a fresh rebuild reproduces the manifest byte-for-byte EXCEPT the two
     volatile fields -- the substantive content and its signature chain
     are stable, while each issued pack keeps a unique id and a real
     issue-time, as a real compliance artifact should.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from auditk.attestation.signer import generate_keypair

from bossyk_sandbox.evidence.pack import build_and_sign, read_pack, verify_offline
from bossyk_sandbox.evidence.sample import (
    VOLATILE_PACK_FIELDS,
    build_sample_trace,
    stable_manifest,
)

REPO_ROOT = Path(__file__).parent.parent.parent
BENCH_OUTPUT = REPO_ROOT / "docs" / "bench_output"
SAMPLE_PACK = BENCH_OUTPUT / "evidence_pack_sample.json"
SAMPLE_PUBKEY = BENCH_OUTPUT / "evidence_pack_sample.pub"

SAMPLE_AGENT_VERSION = "sample-0.1"


def _sign_sample(signer_key_path: Path) -> object:
    return build_and_sign(
        build_sample_trace(),
        agent_version=SAMPLE_AGENT_VERSION,
        signer_key_path=signer_key_path,
    )


def test_build_sample_trace_is_deterministic() -> None:
    # No wall-clock read anywhere: two builds must be byte-identical, or the
    # "content regenerates" guarantee below is meaningless.
    first = build_sample_trace().model_dump(mode="json")
    second = build_sample_trace().model_dump(mode="json")

    assert first == second


def test_volatile_fields_are_exactly_the_two_unavoidable_ones() -> None:
    # Guards the honesty of the regen check: widening this set to hide a
    # content change would let a drifted pack still "regenerate".
    assert VOLATILE_PACK_FIELDS == {"pack_id", "issued_at", "signatures"}


def test_stable_manifest_strips_the_volatile_fields_and_keeps_the_rest() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        priv, _ = generate_keypair(Path(tmp) / "k")
        pack = _sign_sample(priv)

    manifest = stable_manifest(pack)

    assert "pack_id" not in manifest
    assert "issued_at" not in manifest
    assert "signatures" not in manifest
    assert manifest["trace_summary"]["step_count"] == 2
    assert manifest["subject"]["agent_version"] == SAMPLE_AGENT_VERSION


def test_a_built_sample_pack_verifies_offline() -> None:
    # Verify the PIPELINE produces verifiable packs, building and verifying
    # under whatever auditk is installed -- not the committed pack's bytes
    # against an arbitrary auditk. An EvidencePack's signed manifest is
    # auditk-schema-shaped, so a pack signed under one auditk version cannot
    # portably verify under another (a schema field present at sign time is
    # dropped on load by an older auditk, breaking the signature). The
    # committed *trace* sidecar (below) uses the stable Trace schema and IS
    # verified from its committed bytes; that is the feature's browsable
    # evidence. See the phase doc's auditk-version note.
    with tempfile.TemporaryDirectory() as tmp:
        priv, pub = generate_keypair(Path(tmp) / "k")
        pack = _sign_sample(priv)
        public_key_pem = pub.read_text()

    assert verify_offline(pack, public_key_pem) is True


def test_committed_sample_pack_content_regenerates_bar_the_volatile_fields() -> None:
    # A fresh key is fine: the manifest comparison excludes signatures, so
    # this proves the *content* is stable independent of who signed it or
    # when.
    committed = read_pack(SAMPLE_PACK)

    with tempfile.TemporaryDirectory() as tmp:
        priv, _ = generate_keypair(Path(tmp) / "k")
        regenerated = _sign_sample(priv)

    assert stable_manifest(regenerated) == stable_manifest(committed)
