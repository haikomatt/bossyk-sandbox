"""The compliance framework catalogue is a single source of truth,
story-independent: `load_frameworks()` reads the committed
`compliance/frameworks.yaml` so the story pipeline AND the evidence-pack
control tagger (a later phase) resolve control_refs against the same
taxonomy, without either one owning it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from bossyk_sandbox.compliance.frameworks import (
    FrameworkType,
    load_frameworks,
)


def _control(id: str = "art-12", ref: str = "Art. 12", title: str = "Logging") -> dict[str, Any]:
    return {"id": id, "ref": ref, "title": title}


def _framework(
    *,
    id: str = "eu-ai-act",
    name: str = "EU AI Act",
    type: str = "regulation",
    controls: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "id": id,
        "name": name,
        "type": type,
        "controls": controls if controls is not None else [_control()],
    }


def _registry(
    *, entries: list[dict[str, Any]] | None = None, disclaimer: str = "Not legal advice."
) -> dict[str, Any]:
    return {"disclaimer": disclaimer, "entries": entries if entries is not None else [_framework()]}


def _write(tmp_path: Path, registry: dict[str, Any]) -> Path:
    path = tmp_path / "frameworks.yaml"
    path.write_text(yaml.safe_dump(registry))
    return path


# --- the committed package catalogue ----------------------------------


def test_committed_catalogue_loads_with_the_seven_frameworks() -> None:
    # Test-Integrity note: bossyk-sandbox slice 2, P7 added the "pecr"
    # framework (PECR reg 21, outreach boundary 1) alongside the existing
    # six -- this assumption is deliberately invalidated, not silently
    # broken; see compliance/frameworks.yaml and attribution.py's
    # _ACTION_CONTROLS outreach entries.
    registry = load_frameworks()

    ids = [framework.id for framework in registry.entries]
    assert ids == ["eu-ai-act", "fca", "pecr", "hipaa", "iso-42001", "iso-27001", "soc2"]
    assert "not legal advice" in registry.disclaimer.lower()
    # every control the story's substrate/verdict tags will emit must resolve
    resolvable = {f"{f.id}:{c.id}" for f in registry.entries for c in f.controls}
    for ref in ("eu-ai-act:art-12", "soc2:cc7-3", "hipaa:audit-controls", "iso-27001:a-8-15"):
        assert ref in resolvable


def test_committed_catalogue_control_ids_are_unique_per_framework() -> None:
    registry = load_frameworks()
    for framework in registry.entries:
        ids = [control.id for control in framework.controls]
        assert len(ids) == len(set(ids)), f"{framework.id} has duplicate control ids"


# --- load_frameworks over an explicit path ----------------------------


def test_load_frameworks_reads_an_explicit_path(tmp_path: Path) -> None:
    path = _write(tmp_path, _registry(entries=[_framework(id="soc2", type="attestation")]))

    registry = load_frameworks(path)

    assert registry.entries[0].id == "soc2"
    assert registry.entries[0].type == FrameworkType.ATTESTATION


def test_duplicate_framework_ids_raise(tmp_path: Path) -> None:
    path = _write(tmp_path, _registry(entries=[_framework(id="soc2"), _framework(id="soc2")]))

    with pytest.raises(ValueError, match="soc2"):
        load_frameworks(path)


def test_duplicate_control_ids_within_a_framework_raise(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        _registry(entries=[_framework(controls=[_control("art-12"), _control("art-12")])]),
    )

    with pytest.raises(ValueError, match="art-12"):
        load_frameworks(path)


def test_framework_and_control_ids_must_be_kebab_slugs(tmp_path: Path) -> None:
    path = _write(tmp_path, _registry(entries=[_framework(id="EU_AI_Act")]))

    with pytest.raises(ValueError):
        load_frameworks(path)


def test_unknown_framework_type_raises(tmp_path: Path) -> None:
    path = _write(tmp_path, _registry(entries=[_framework(type="guideline")]))

    with pytest.raises(ValueError):
        load_frameworks(path)
