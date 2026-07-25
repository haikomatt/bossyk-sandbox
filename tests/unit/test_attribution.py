"""The evidence-pack control tagger (Tiers A + B): which compliance
controls a single attested action discharges, purely from what is known at
attestation time -- that it is a gated, attested step (substrate) and its
verdict/override (verdict-derived). Each tag carries a `basis` stating why
it applies, the honesty guardrail carried down from the story coverage
work. Tier C (boundary / PII specific) needs a new signal and is a later
phase; nothing here reads the tool name.
"""

from __future__ import annotations

from bossyk_sandbox.compliance.attribution import (
    ControlTag,
    controls_for_step,
)
from bossyk_sandbox.compliance.frameworks import load_frameworks
from bossyk_sandbox.instruments.base import Verdict


def _refs(tags: list[ControlTag]) -> list[str]:
    return [tag.ref for tag in tags]


def _by_basis(tags: list[ControlTag]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for tag in tags:
        out.setdefault(tag.basis, []).append(tag.ref)
    return out


def test_every_attested_step_carries_the_substrate_controls() -> None:
    # No matter the verdict, the record-keeping controls apply, because the
    # step IS a signed, gated log entry.
    for verdict in (Verdict.ALLOW, Verdict.BLOCK):
        for overridden in (False, True):
            substrate = _by_basis(controls_for_step(verdict, overridden=overridden)).get(
                "substrate", []
            )
            assert set(substrate) == {
                "eu-ai-act:art-12",
                "iso-27001:a-8-15",
                "iso-42001:records",
                "soc2:cc7-2",
                "hipaa:audit-controls",
            }


def test_an_allowed_unoverridden_step_is_substrate_plus_gated_only() -> None:
    tags = controls_for_step(Verdict.ALLOW, overridden=False)
    by_basis = _by_basis(tags)

    assert set(by_basis) == {"substrate", "verdict:gated"}
    assert set(by_basis["verdict:gated"]) == {"eu-ai-act:art-9", "soc2:cc7-3"}


def test_a_blocked_step_adds_the_incident_response_controls() -> None:
    tags = controls_for_step(Verdict.BLOCK, overridden=False)
    by_basis = _by_basis(tags)

    assert set(by_basis["verdict:blocked"]) == {"soc2:cc7-4", "fca:sysc"}
    # blocked is not overridden here
    assert "verdict:overridden" not in by_basis


def test_an_overridden_step_adds_the_human_oversight_control() -> None:
    tags = controls_for_step(Verdict.ALLOW, overridden=True)
    by_basis = _by_basis(tags)

    assert by_basis["verdict:overridden"] == ["eu-ai-act:art-14"]


def test_tags_are_deterministic_and_free_of_duplicate_refs() -> None:
    tags = controls_for_step(Verdict.BLOCK, overridden=True)
    refs = _refs(tags)

    assert refs == _refs(controls_for_step(Verdict.BLOCK, overridden=True))  # stable
    assert len(refs) == len(set(refs))  # no dup refs across tiers
    # substrate leads, then gated, then blocked, then overridden
    assert refs[0].startswith("eu-ai-act:art-12")


def test_every_emitted_ref_resolves_against_the_committed_catalogue() -> None:
    registry = load_frameworks()
    resolvable = {f"{f.id}:{c.id}" for f in registry.entries for c in f.controls}

    seen: set[str] = set()
    for verdict in (Verdict.ALLOW, Verdict.BLOCK):
        for overridden in (False, True):
            for tag in controls_for_step(verdict, overridden=overridden):
                seen.add(tag.ref)
    assert seen, "expected some tags"
    unresolved = seen - resolvable
    assert unresolved == set(), f"tagger emits refs not in the catalogue: {unresolved}"
