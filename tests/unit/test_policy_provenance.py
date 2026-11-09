"""Policy provenance binding (Phase A).

A signature proves a trace was not altered since signing. It does not say
WHICH policy the gate was running. These tests pin the second claim: a trace
carries a manifest of the policy that actually ran (policy YAML hash, the
instantiated fast rules, the standing grants), and a verifier checks it
against an approved manifest.

Shape copied from `test_attestation_tcb_forgery.py`: one check passes
(signature), the other fails (provenance), against the same key.

No network, no spend: the policy instrument is built with a dummy key and is
never asked to score.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from pathlib import Path

import pytest
from auditk.attestation.signer import generate_keypair
from auditk.schema import Trace

from bossyk_sandbox.domains import DomainConfig, domain_config
from bossyk_sandbox.evidence.policy_provenance import (
    MANIFEST_METADATA_KEY,
    DiffKind,
    ManifestSerialisationError,
    PolicyManifest,
    ProvenanceNote,
    ProvenanceStatus,
    bind_policy_manifest,
    manifest_for,
    verify_policy_provenance,
)
from bossyk_sandbox.evidence.signed_trace import (
    SignedTrace,
    build_signed_trace,
    verify_signed_trace,
)
from bossyk_sandbox.evidence.trace import build_trace, make_step
from bossyk_sandbox.instruments.base import Decision, Instrument, ProposedAction, Verdict
from bossyk_sandbox.instruments.hardcoded_rule import RequirePassedCheck
from bossyk_sandbox.instruments.policy import build_default_policy_instrument
from bossyk_sandbox.standing import StandingGrant, retail_standing_grants
from tests.unit._bossyk import SYNTHETIC_POLICY, FakeBossyk, requires_bossyk_checkout


def _trace() -> Trace:
    call = ProposedAction("get_order_details", {"order_id": "#W1"})
    step = make_step("t1", call, Decision(Verdict.ALLOW, "ok"), step_id="s1")
    return build_trace("t1", "retail-test@0.1", [step])


def _manifest(
    cfg: DomainConfig,
    *,
    fast_rules: list[Instrument] | None = None,
    grants: dict[str, StandingGrant] | None = None,
    policy_bytes: bytes | None = None,
) -> PolicyManifest:
    return manifest_for(
        cfg,
        fast_rules if fast_rules is not None else cfg.fast_rules_factory(),
        grants,
        policy_bytes if policy_bytes is not None else SYNTHETIC_POLICY,
    )


def _signed(manifest: PolicyManifest | None, key_path: Path) -> SignedTrace:
    trace = _trace()
    if manifest is not None:
        bind_policy_manifest(trace, manifest)
    return build_signed_trace(trace, signer_key_path=key_path)


@pytest.fixture
def keys(tmp_path: Path) -> tuple[Path, str]:
    private, public = generate_keypair(tmp_path / "signer")
    return private, public.read_text()


def test_manifest_is_deterministic() -> None:
    cfg = domain_config("retail")
    a = _manifest(cfg, grants=retail_standing_grants())
    b = _manifest(cfg, grants=retail_standing_grants())
    assert a.manifest_sha256 == b.manifest_sha256
    assert a == b


def test_matching_config_passes_signature_and_provenance(keys: tuple[Path, str]) -> None:
    private, public = keys
    cfg = domain_config("retail")
    approved = _manifest(cfg, grants=retail_standing_grants())
    signed = _signed(approved, private)
    assert verify_signed_trace(signed, public) is True
    report = verify_policy_provenance(signed, approved)
    assert report.status is ProvenanceStatus.MATCH
    assert report.diffs == []


def test_loosened_fast_rule_verifies_by_signature_but_fails_provenance(
    keys: tuple[Path, str],
) -> None:
    private, public = keys
    cfg = domain_config("retail")
    approved = _manifest(cfg, grants=retail_standing_grants())

    loosened_rules = [
        r
        for r in cfg.fast_rules_factory()
        if getattr(r, "gated_tool", None) != "cancel_pending_order"
    ]
    assert len(loosened_rules) == len(cfg.fast_rules_factory()) - 1
    loosened = _manifest(cfg, fast_rules=loosened_rules, grants=retail_standing_grants())
    signed = _signed(loosened, private)

    assert verify_signed_trace(signed, public) is True  # signature is clean
    report = verify_policy_provenance(signed, approved)
    assert report.status is ProvenanceStatus.MISMATCH
    removed = [d for d in report.diffs if d.kind is DiffKind.RULE_REMOVED]
    assert len(removed) == 1
    assert "cancel_pending_order" in removed[0].detail


def test_loosened_grant_verifies_by_signature_but_fails_provenance(
    keys: tuple[Path, str],
) -> None:
    private, public = keys
    cfg = domain_config("retail")
    approved = _manifest(cfg, grants=retail_standing_grants())

    grants = retail_standing_grants()
    grants["refund"] = dataclasses.replace(grants["refund"], max_count=3)
    signed = _signed(_manifest(cfg, grants=grants), private)

    assert verify_signed_trace(signed, public) is True
    report = verify_policy_provenance(signed, approved)
    assert report.status is ProvenanceStatus.MISMATCH
    changed = [d for d in report.diffs if d.kind is DiffKind.GRANT_FIELD_CHANGED]
    assert len(changed) == 1
    assert changed[0].path == "standing.refund.max_count"
    assert (changed[0].approved, changed[0].actual) == (1, 3)


def test_diff_does_not_judge_direction(keys: tuple[Path, str]) -> None:
    """Design rule 4: report WHAT changed, never loosened/tightened."""
    private, _ = keys
    cfg = domain_config("retail")
    approved = _manifest(cfg, grants=retail_standing_grants())
    grants = retail_standing_grants()
    grants["refund"] = dataclasses.replace(grants["refund"], max_count=0)
    report = verify_policy_provenance(_signed(_manifest(cfg, grants=grants), private), approved)
    assert report.status is ProvenanceStatus.MISMATCH
    for d in report.diffs:
        text = f"{d.kind.value} {d.detail}".lower()
        assert "loosen" not in text and "tighten" not in text


def test_trace_without_manifest_is_unbound(keys: tuple[Path, str]) -> None:
    private, public = keys
    approved = _manifest(domain_config("retail"), grants=retail_standing_grants())
    signed = _signed(None, private)
    assert verify_signed_trace(signed, public) is True
    report = verify_policy_provenance(signed, approved)
    assert report.status is ProvenanceStatus.UNBOUND  # UNBOUND is never MATCH


def test_malformed_manifest_is_unbound(keys: tuple[Path, str]) -> None:
    private, _ = keys
    trace = _trace()
    trace.metadata[MANIFEST_METADATA_KEY] = {"garbage": True}
    signed = build_signed_trace(trace, signer_key_path=private)
    approved = _manifest(domain_config("retail"), grants=retail_standing_grants())
    assert verify_policy_provenance(signed, approved).status is ProvenanceStatus.UNBOUND


def test_manifest_tampered_after_signing_breaks_signature(keys: tuple[Path, str]) -> None:
    """Binding is covered by the existing signature: swapping in the approved
    manifest after signing a loosened run invalidates the signature."""
    private, public = keys
    cfg = domain_config("retail")
    approved = _manifest(cfg, grants=retail_standing_grants())
    grants = retail_standing_grants()
    grants["refund"] = dataclasses.replace(grants["refund"], max_count=3)
    signed = _signed(_manifest(cfg, grants=grants), private)
    signed.trace.metadata[MANIFEST_METADATA_KEY] = approved.model_dump(mode="json")
    assert verify_signed_trace(signed, public) is False


def test_yaml_byte_change_gives_yaml_hash_diff(keys: tuple[Path, str]) -> None:
    private, public = keys
    cfg = domain_config("retail")
    approved = _manifest(cfg, grants=retail_standing_grants())
    edited = SYNTHETIC_POLICY + b"\n# one extra byte-level edit\n"
    signed = _signed(_manifest(cfg, grants=retail_standing_grants(), policy_bytes=edited), private)
    assert verify_signed_trace(signed, public) is True
    report = verify_policy_provenance(signed, approved)
    assert report.status is ProvenanceStatus.MISMATCH
    assert [d.kind for d in report.diffs] == [DiffKind.YAML_HASH_CHANGED]
    assert report.diffs[0].path == "policy_yaml_sha256"


def test_airline_without_grants_matches_with_standing_not_configured(
    keys: tuple[Path, str],
) -> None:
    private, _ = keys
    approved = _manifest(domain_config("airline"), grants=None)
    assert approved.standing is None  # explicit null, not omitted
    report = verify_policy_provenance(_signed(approved, private), approved)
    assert report.status is ProvenanceStatus.MATCH
    assert ProvenanceNote.STANDING_NOT_CONFIGURED in report.notes


def test_empty_grants_also_record_null_standing() -> None:
    assert _manifest(domain_config("advice"), grants={}).standing is None


def test_retail_match_has_no_standing_note(keys: tuple[Path, str]) -> None:
    private, _ = keys
    approved = _manifest(domain_config("retail"), grants=retail_standing_grants())
    report = verify_policy_provenance(_signed(approved, private), approved)
    assert ProvenanceNote.STANDING_NOT_CONFIGURED not in report.notes


class _OpaqueRule:
    """Not a dataclass, state is a live object: cannot be canonically serialised."""

    def __init__(self) -> None:
        self.handle = object()

    def score(self, proposed: ProposedAction, history: object) -> Decision:
        return Decision(Verdict.ALLOW, "x")


def test_unserialisable_rule_state_is_refused() -> None:
    cfg = domain_config("airline")
    with pytest.raises(ManifestSerialisationError):
        _manifest(cfg, fast_rules=[_OpaqueRule()])


def test_lambda_predicate_is_refused() -> None:
    cfg = domain_config("outreach")
    rule = RequirePassedCheck("book_survey", "check_eligibility", "prospect_id", lambda r: True)
    with pytest.raises(ManifestSerialisationError):
        _manifest(cfg, fast_rules=[rule])


def _predicate_descriptors(rule: object) -> list[dict[str, object]]:
    """Every {module, qualname, source_sha256} descriptor in a manifest's rules."""
    found: list[dict[str, object]] = []

    def walk(node: object) -> None:
        if isinstance(node, dict):
            if "source_sha256" in node:
                found.append(node)
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(rule)
    return found


def test_both_outreach_predicates_serialise_as_module_qualname_source_hash() -> None:
    import hashlib
    import inspect

    from bossyk_sandbox.scenarios import runner

    manifest = _manifest(domain_config("outreach"))
    descriptors = _predicate_descriptors(
        [r.model_dump(mode="json") if hasattr(r, "model_dump") else r for r in manifest.fast_rules]
    )
    assert len(descriptors) == 2
    by_name = {d["qualname"]: d for d in descriptors}
    for fn in (runner._outreach_contact_is_clean, runner._outreach_prospect_is_eligible):
        d = by_name[fn.__qualname__]
        assert d["module"] == fn.__module__
        assert d["source_sha256"] == hashlib.sha256(inspect.getsource(fn).encode()).hexdigest()
    assert _manifest(domain_config("outreach")).manifest_sha256 == manifest.manifest_sha256


def _rule(pred: object) -> RequirePassedCheck:
    return RequirePassedCheck("g", "c", "k", pred)  # type: ignore[arg-type]


class _Holder:
    def method(self, result: object) -> bool:
        return True


def test_closure_predicate_is_refused() -> None:
    def make(flag: bool) -> Callable[[object], bool]:
        def pred(result: object) -> bool:
            return bool(result) is flag

        return pred

    with pytest.raises(ManifestSerialisationError):
        _manifest(domain_config("outreach"), fast_rules=[_rule(make(True))])


def test_partial_predicate_is_refused() -> None:
    import functools

    with pytest.raises(ManifestSerialisationError):
        _manifest(
            domain_config("outreach"),
            fast_rules=[_rule(functools.partial(_top_level_true))],
        )


def test_bound_method_predicate_is_refused() -> None:
    with pytest.raises(ManifestSerialisationError):
        _manifest(domain_config("outreach"), fast_rules=[_rule(_Holder().method)])


def test_callable_object_predicate_is_refused() -> None:
    class Callish:
        def __call__(self, result: object) -> bool:
            return True

    with pytest.raises(ManifestSerialisationError):
        _manifest(domain_config("outreach"), fast_rules=[_rule(Callish())])


def test_predicate_without_retrievable_source_is_refused() -> None:
    ns: dict[str, object] = {}
    exec("def pred(result):\n    return True\n", ns)  # noqa: S102 - no file, no source
    with pytest.raises(ManifestSerialisationError):
        _manifest(domain_config("outreach"), fast_rules=[_rule(ns["pred"])])


def _top_level_true(result: object) -> bool:
    return True


def _top_level_false(result: object) -> bool:
    return False


def test_different_named_predicates_differ() -> None:
    cfg = domain_config("outreach")
    a = _manifest(cfg, fast_rules=[RequirePassedCheck("g", "c", "k", _top_level_true)])
    b = _manifest(cfg, fast_rules=[RequirePassedCheck("g", "c", "k", _top_level_false)])
    assert a.manifest_sha256 != b.manifest_sha256


# --- design rule 1: hash what you load, at load time ---------------------


def test_policy_instrument_binds_hash_of_loaded_bytes_not_the_path(
    tmp_path: Path, fake_bossyk: FakeBossyk
) -> None:
    import hashlib

    original = SYNTHETIC_POLICY
    policy_file = tmp_path / "policy.yaml"
    policy_file.write_bytes(original)

    instrument = build_default_policy_instrument(policy_file)

    # Edit the file on disk AFTER load: the bound hash must not move.
    policy_file.write_bytes(original.replace(b"policy_id:", b"policy_id: tampered-", 1))

    assert instrument.policy_sha256 == hashlib.sha256(original).hexdigest()
    assert instrument.policy_bytes == original
    # and the judge actually parsed the original bytes, not the edited file
    assert "tampered" not in instrument.judge._policy.policy_id  # type: ignore[attr-defined]


def test_policy_instrument_snapshot_is_not_the_original_path(
    tmp_path: Path, fake_bossyk: FakeBossyk
) -> None:
    """load_policy must never re-read the original path (TOCTOU)."""
    policy_file = tmp_path / "policy.yaml"
    policy_file.write_bytes(SYNTHETIC_POLICY)
    build_default_policy_instrument(policy_file)
    assert fake_bossyk.loaded_paths
    assert all(p != policy_file for p in fake_bossyk.loaded_paths)


@requires_bossyk_checkout
def test_policy_instrument_with_the_real_bossyk_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same hash-of-loaded-bytes property against bossyk's REAL
    `load_policy` / `PolicyAwareJudge` (dev machines only; skipped in CI)."""
    import hashlib

    monkeypatch.setenv("FIREWORKS_API_KEY", "dummy-never-used")  # never scored
    original = domain_config("retail").policy_path.read_bytes()
    policy_file = tmp_path / "policy.yaml"
    policy_file.write_bytes(original)
    instrument = build_default_policy_instrument(policy_file)
    policy_file.write_bytes(original + b"\n# edited after load\n")
    assert instrument.policy_sha256 == hashlib.sha256(original).hexdigest()
