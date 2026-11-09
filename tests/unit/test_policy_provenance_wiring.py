"""Regression ratchet: design rule 1 bypass in the call-site wiring.

Phase A Green's call sites used `manifest_for_domain(domain, instruments,
grants)` with a disk fallback, so the YAML hash was taken from
`cfg.policy_path.read_bytes()` when the trace was BUILT, not from the bytes a
`PolicyInstrument` loaded. That is a TOCTOU bypass, and when no policy
instrument ran at all the manifest still claimed a hash for a policy that
never loaded.

Guard: policy bytes come only from the `PolicyInstrument` among the gate's
instruments. No instrument -> `policy_yaml_sha256` is None and the verifier
reports `POLICY_NOT_LOADED`. The disk is never read.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from auditk.attestation.signer import generate_keypair

from bossyk_sandbox.conditions.live_replay import _live_run_result
from bossyk_sandbox.domains import domain_config
from bossyk_sandbox.evidence.policy_provenance import (
    MANIFEST_METADATA_KEY,
    DiffKind,
    ManifestSerialisationError,
    PolicyManifest,
    ProvenanceNote,
    ProvenanceStatus,
    manifest_for,
    manifest_for_domain,
    verify_policy_provenance,
)
from bossyk_sandbox.evidence.signed_trace import build_signed_trace
from bossyk_sandbox.gate import Gate
from bossyk_sandbox.instruments.policy import build_default_policy_instrument
from bossyk_sandbox.scenarios.runner import retail_fast_rules


def _loaded_instrument(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str = "p.yaml"):  # type: ignore[no-untyped-def]
    monkeypatch.setenv("FIREWORKS_API_KEY", "dummy-never-used")
    original = domain_config("retail").policy_path.read_bytes()
    policy_file = tmp_path / name
    policy_file.write_bytes(original)
    return build_default_policy_instrument(policy_file), policy_file, original


def _stub_session() -> Any:
    return SimpleNamespace(
        steps=[],
        gate=Gate(instruments=retail_fast_rules()),
        tool_latency=[],
        agent_latency=[],
        utterance_decisions=[],
    )


def test_call_site_binds_hash_of_loaded_bytes_not_current_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    instrument, policy_file, original = _loaded_instrument(tmp_path, monkeypatch)
    # The operator edits the policy AFTER the instrument loaded it, BEFORE the trace is built.
    policy_file.write_bytes(original + b"\n# loosened after load\n")

    result = _live_run_result(
        _stub_session(), "t1", "ref@0.1", domain="retail", slow_instruments=[instrument]
    )

    bound = PolicyManifest.model_validate(result.trace.metadata[MANIFEST_METADATA_KEY])
    assert bound.policy_yaml_sha256 == hashlib.sha256(original).hexdigest()


def test_call_site_without_policy_instrument_records_none_and_never_reads_disk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(self: Path) -> bytes:
        raise AssertionError("policy file must not be read at trace-build time")

    monkeypatch.setattr(Path, "read_bytes", boom)
    result = _live_run_result(_stub_session(), "t1", "ref@0.1", domain="retail")
    bound = PolicyManifest.model_validate(result.trace.metadata[MANIFEST_METADATA_KEY])
    assert bound.policy_yaml_sha256 is None


def test_manifest_for_domain_takes_bytes_only_from_the_instrument(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    instrument, policy_file, original = _loaded_instrument(tmp_path, monkeypatch)
    policy_file.write_bytes(b"policy_id: swapped\n")
    manifest = manifest_for_domain(
        "retail", retail_fast_rules(), None, slow_instruments=[instrument]
    )
    assert manifest.policy_yaml_sha256 == hashlib.sha256(original).hexdigest()
    assert instrument.__class__.__name__ not in repr(manifest.fast_rules)  # not described as a rule


def test_two_policy_instruments_are_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    a, _, _ = _loaded_instrument(tmp_path, monkeypatch, "a.yaml")
    b, _, _ = _loaded_instrument(tmp_path, monkeypatch, "b.yaml")
    with pytest.raises(ManifestSerialisationError):
        manifest_for_domain("retail", retail_fast_rules(), None, slow_instruments=[a, b])


def test_not_loaded_policy_reports_note_and_diffs_against_a_hashed_approval(
    tmp_path: Path,
) -> None:
    private, _ = generate_keypair(tmp_path / "k")
    cfg = domain_config("retail")
    approved = manifest_for(cfg, cfg.fast_rules_factory(), None, cfg.policy_path.read_bytes())
    unloaded = manifest_for_domain("retail", cfg.fast_rules_factory(), None)
    assert unloaded.policy_yaml_sha256 is None

    result = _live_run_result(_stub_session(), "t1", "ref@0.1", domain="retail")
    signed = build_signed_trace(result.trace, signer_key_path=private)
    report = verify_policy_provenance(signed, approved)
    assert report.status is ProvenanceStatus.MISMATCH
    assert ProvenanceNote.POLICY_NOT_LOADED in report.notes
    yaml_diffs = [d for d in report.diffs if d.kind is DiffKind.YAML_HASH_CHANGED]
    assert len(yaml_diffs) == 1 and yaml_diffs[0].actual is None


def test_schema_version_was_bumped_for_nullable_policy_hash() -> None:
    from bossyk_sandbox.evidence.policy_provenance import MANIFEST_SCHEMA_VERSION

    assert MANIFEST_SCHEMA_VERSION >= 2
