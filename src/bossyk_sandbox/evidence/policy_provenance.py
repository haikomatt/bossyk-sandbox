"""Policy provenance: bind the policy the gate actually ran into the signed trace.

A signature proves a trace was not altered since signing. It says nothing
about WHICH policy the gate was running, so a loosened rule set signs just as
cleanly as the approved one. This module closes that gap in two halves:

* ``manifest_for`` / ``bind_policy_manifest`` record a ``PolicyManifest`` --
  the policy YAML hash, the instantiated fast rules and the standing grants --
  in ``Trace.metadata`` under a namespaced key. ``Trace.metadata`` sits inside
  the signed manifest, so the existing signature covers the binding.
* ``verify_policy_provenance`` compares a signed trace's bound manifest with
  an approved one and reports a field-level diff.

Design rules (scope doc, fixed):

* The fast-rule part is derived from the INSTANTIATED rule objects, never the
  factory name. State that cannot be canonically serialised raises
  ``ManifestSerialisationError`` rather than falling back to ``repr()``.
* A trace with no (or a malformed) manifest is ``UNBOUND``, never a match.
* The diff says WHAT changed, never whether it loosened or tightened.
* "No standing configured" is explicit: ``standing`` is ``None`` and the
  report carries ``STANDING_NOT_CONFIGURED``.

The manifest is still declared by the runtime: see
``probes/attestation/tcb_claim_boundary.md``. Bossyk-side only (demonstrator
section 13): not proposed to auditk-spec.
"""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from auditk.attestation.canonical import canonicalize
from auditk.schema import Trace
from pydantic import BaseModel, ValidationError

from bossyk_sandbox.domains import DomainConfig, domain_config
from bossyk_sandbox.evidence.signed_trace import SignedTrace
from bossyk_sandbox.instruments.base import Instrument
from bossyk_sandbox.instruments.policy import PolicyInstrument
from bossyk_sandbox.standing import StandingGrant

MANIFEST_METADATA_KEY = "bossyk.policy_manifest"
# v2: `policy_yaml_sha256` is nullable (no policy instrument loaded the YAML).
MANIFEST_SCHEMA_VERSION = 2


class ManifestSerialisationError(ValueError):
    """A fast rule carries state the manifest cannot canonically serialise."""


class PolicyManifest(BaseModel):
    schema_version: int = MANIFEST_SCHEMA_VERSION
    domain: str
    policy_yaml_sha256: str | None
    fast_rules: list[dict[str, Any]]
    standing: dict[str, dict[str, Any]] | None
    manifest_sha256: str


class ProvenanceStatus(StrEnum):
    MATCH = "MATCH"
    MISMATCH = "MISMATCH"
    UNBOUND = "UNBOUND"


class ProvenanceNote(StrEnum):
    STANDING_NOT_CONFIGURED = "STANDING_NOT_CONFIGURED"
    POLICY_NOT_LOADED = "POLICY_NOT_LOADED"


class DiffKind(StrEnum):
    DOMAIN_CHANGED = "DOMAIN_CHANGED"
    YAML_HASH_CHANGED = "YAML_HASH_CHANGED"
    RULE_REMOVED = "RULE_REMOVED"
    RULE_ADDED = "RULE_ADDED"
    RULE_PARAMS_CHANGED = "RULE_PARAMS_CHANGED"
    STANDING_CHANGED = "STANDING_CHANGED"
    GRANT_REMOVED = "GRANT_REMOVED"
    GRANT_ADDED = "GRANT_ADDED"
    GRANT_FIELD_CHANGED = "GRANT_FIELD_CHANGED"


@dataclass(frozen=True)
class ManifestDiff:
    kind: DiffKind
    path: str
    approved: Any
    actual: Any
    detail: str


@dataclass(frozen=True)
class ProvenanceReport:
    status: ProvenanceStatus
    diffs: list[ManifestDiff] = field(default_factory=list)
    notes: set[ProvenanceNote] = field(default_factory=set)


# --- serialisation ----------------------------------------------------


def _describe_predicate(fn: Any, where: str) -> dict[str, str]:
    """A predicate is identified by {module, qualname, source_sha256}. Source
    text, not bytecode: bytecode varies across Python versions. Only plain
    module-level functions qualify; anything else cannot be identified by name
    plus source and is refused."""
    if not inspect.isfunction(fn):
        raise ManifestSerialisationError(
            f"{where}: {type(fn).__name__} is not a plain module-level function"
        )
    if fn.__name__ == "<lambda>":
        raise ManifestSerialisationError(f"{where}: lambdas cannot be identified by name")
    if fn.__closure__ is not None or "<locals>" in fn.__qualname__:
        raise ManifestSerialisationError(f"{where}: closures/nested functions are refused")
    try:
        source = inspect.getsource(fn)
    except (OSError, TypeError) as exc:
        raise ManifestSerialisationError(f"{where}: source unavailable ({exc})") from exc
    return {
        "module": fn.__module__,
        "qualname": fn.__qualname__,
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
    }


def _serialise(value: Any, where: str) -> Any:
    if value is None or isinstance(value, bool | int | float | str):
        return value.value if isinstance(value, StrEnum) else value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            f.name: _serialise(getattr(value, f.name), f"{where}.{f.name}")
            for f in dataclasses.fields(value)
        }
    if isinstance(value, list | tuple):
        return [_serialise(v, f"{where}[]") for v in value]
    if isinstance(value, Mapping):
        return {str(k): _serialise(v, f"{where}.{k}") for k, v in value.items()}
    if callable(value):
        return _describe_predicate(value, where)
    raise ManifestSerialisationError(
        f"{where}: cannot canonically serialise {type(value).__name__}"
    )


def _describe_rule(rule: Instrument) -> dict[str, Any]:
    cls = type(rule)
    where = cls.__qualname__
    if not dataclasses.is_dataclass(rule):
        raise ManifestSerialisationError(f"{where}: not a dataclass, state cannot be serialised")
    return {
        "type": f"{cls.__module__}.{cls.__qualname__}",
        "params": _serialise(rule, where),
    }


def _describe_standing(
    grants: Mapping[str, StandingGrant] | None,
) -> dict[str, dict[str, Any]] | None:
    if not grants:
        return None
    return {name: _serialise(grant, f"standing.{name}") for name, grant in sorted(grants.items())}


def _digest(body: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonicalize(dict(body))).hexdigest()


def manifest_for(
    domain_cfg: DomainConfig,
    fast_rules: Sequence[Instrument],
    grants: Mapping[str, StandingGrant] | None,
    policy_bytes: bytes | None,
) -> PolicyManifest:
    body: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "domain": domain_cfg.name,
        "policy_yaml_sha256": (
            hashlib.sha256(policy_bytes).hexdigest() if policy_bytes is not None else None
        ),
        "fast_rules": [_describe_rule(r) for r in fast_rules],
        "standing": _describe_standing(grants),
    }
    return PolicyManifest(**body, manifest_sha256=_digest(body))


def manifest_for_domain(
    domain: str,
    instruments: Sequence[Any],
    grants: Mapping[str, StandingGrant] | None,
    *,
    slow_instruments: Sequence[Any] = (),
) -> PolicyManifest:
    """Manifest for call sites that know the domain name and the instruments
    the gate was handed. Policy bytes come ONLY from the `PolicyInstrument`
    among `instruments` + `slow_instruments` (its `policy_bytes`, read once at
    load). There is deliberately no disk fallback: re-reading the file at
    trace-build time would be a TOCTOU bypass of design rule 1. With no
    policy instrument the manifest records `policy_yaml_sha256: None`.
    More than one policy instrument is ambiguous and is refused."""
    loaded = [i for i in (*instruments, *slow_instruments) if isinstance(i, PolicyInstrument)]
    if len(loaded) > 1:
        raise ManifestSerialisationError(
            f"{len(loaded)} policy instruments found; the loaded policy is ambiguous"
        )
    rules = [i for i in instruments if not isinstance(i, PolicyInstrument)]
    return manifest_for(
        domain_config(domain), rules, grants, loaded[0].policy_bytes if loaded else None
    )


def bind_policy_manifest(trace: Trace, manifest: PolicyManifest) -> None:
    trace.metadata[MANIFEST_METADATA_KEY] = manifest.model_dump(mode="json")


# --- verification -----------------------------------------------------


def _bound_manifest(trace: Trace) -> PolicyManifest | None:
    """Defensive read: anything absent, malformed or self-inconsistent is None."""
    raw = trace.metadata.get(MANIFEST_METADATA_KEY)
    if not isinstance(raw, dict):
        return None
    try:
        manifest = PolicyManifest.model_validate(raw)
    except ValidationError:
        return None
    body = manifest.model_dump(mode="json", exclude={"manifest_sha256"})
    return manifest if _digest(body) == manifest.manifest_sha256 else None


def _rule_key(rule: Mapping[str, Any]) -> tuple[str, str]:
    params = rule.get("params", {})
    anchor = params.get("gated_tool") or params.get("predicate_tool_name") or ""
    return str(rule.get("type")), str(anchor)


def _canon(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def _diff_rules(approved: list[dict[str, Any]], actual: list[dict[str, Any]]) -> list[ManifestDiff]:
    left = {_canon(r): r for r in approved}
    right = {_canon(r): r for r in actual}
    only_approved = [r for k, r in left.items() if k not in right]
    only_actual = [r for k, r in right.items() if k not in left]
    diffs: list[ManifestDiff] = []
    for rule in only_approved:
        key = _rule_key(rule)
        twin = next((a for a in only_actual if _rule_key(a) == key), None)
        if twin is not None:
            only_actual.remove(twin)
            diffs.append(
                ManifestDiff(
                    DiffKind.RULE_PARAMS_CHANGED,
                    f"fast_rules.{key[0]}[{key[1]}]",
                    rule["params"],
                    twin["params"],
                    f"rule {key[0]} gating {key[1]!r}: params changed",
                )
            )
        else:
            diffs.append(
                ManifestDiff(
                    DiffKind.RULE_REMOVED,
                    f"fast_rules.{key[0]}[{key[1]}]",
                    rule,
                    None,
                    f"rule {key[0]} gating {key[1]!r} removed",
                )
            )
    for rule in only_actual:
        key = _rule_key(rule)
        diffs.append(
            ManifestDiff(
                DiffKind.RULE_ADDED,
                f"fast_rules.{key[0]}[{key[1]}]",
                None,
                rule,
                f"rule {key[0]} gating {key[1]!r} added",
            )
        )
    return diffs


def _diff_standing(
    approved: dict[str, dict[str, Any]] | None, actual: dict[str, dict[str, Any]] | None
) -> list[ManifestDiff]:
    if approved == actual:
        return []
    if approved is None or actual is None:
        return [
            ManifestDiff(
                DiffKind.STANDING_CHANGED,
                "standing",
                approved,
                actual,
                "standing configured on one side only",
            )
        ]
    diffs: list[ManifestDiff] = []
    for boundary in sorted(approved.keys() | actual.keys()):
        a, b = approved.get(boundary), actual.get(boundary)
        if a is None:
            diffs.append(
                ManifestDiff(DiffKind.GRANT_ADDED, f"standing.{boundary}", None, b, "grant added")
            )
        elif b is None:
            diffs.append(
                ManifestDiff(
                    DiffKind.GRANT_REMOVED, f"standing.{boundary}", a, None, "grant removed"
                )
            )
        else:
            for name in sorted(a.keys() | b.keys()):
                if a.get(name) != b.get(name):
                    diffs.append(
                        ManifestDiff(
                            DiffKind.GRANT_FIELD_CHANGED,
                            f"standing.{boundary}.{name}",
                            a.get(name),
                            b.get(name),
                            f"grant {boundary}.{name} changed",
                        )
                    )
    return diffs


def verify_policy_provenance(signed: SignedTrace, approved: PolicyManifest) -> ProvenanceReport:
    """Compare the manifest bound into ``signed`` with ``approved``. Does NOT
    check the signature: call ``verify_signed_trace`` for that."""
    actual = _bound_manifest(signed.trace)
    if actual is None:
        return ProvenanceReport(ProvenanceStatus.UNBOUND)
    diffs: list[ManifestDiff] = []
    if actual.domain != approved.domain:
        diffs.append(
            ManifestDiff(
                DiffKind.DOMAIN_CHANGED,
                "domain",
                approved.domain,
                actual.domain,
                "domain changed",
            )
        )
    if actual.policy_yaml_sha256 != approved.policy_yaml_sha256:
        diffs.append(
            ManifestDiff(
                DiffKind.YAML_HASH_CHANGED,
                "policy_yaml_sha256",
                approved.policy_yaml_sha256,
                actual.policy_yaml_sha256,
                "policy YAML hash changed",
            )
        )
    diffs += _diff_rules(approved.fast_rules, actual.fast_rules)
    diffs += _diff_standing(approved.standing, actual.standing)
    notes: set[ProvenanceNote] = set()
    if actual.standing is None:
        notes.add(ProvenanceNote.STANDING_NOT_CONFIGURED)
    if actual.policy_yaml_sha256 is None:
        notes.add(ProvenanceNote.POLICY_NOT_LOADED)
    status = ProvenanceStatus.MISMATCH if diffs else ProvenanceStatus.MATCH
    return ProvenanceReport(status, diffs, notes)
