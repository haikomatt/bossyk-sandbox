"""The compliance-framework catalogue: the single, story-independent
source of truth for which regulatory controls exist and what they are
called. `compliance/frameworks.yaml` holds the data; `load_frameworks`
reads it. Both the story pipeline (`bossyk_sandbox.story`) and the
evidence-pack control tagger resolve `<framework>:<control>` refs against
this one catalogue, so neither owns the taxonomy.

Lives below the story layer on purpose: story depends on compliance, never
the reverse, so the pack tagger can use the catalogue without dragging in
the whole story schema.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, field_validator, model_validator

_KEBAB_SLUG_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")

_DEFAULT_CATALOGUE = Path(__file__).parent / "frameworks.yaml"


def require_kebab_slug(value: str, label: str) -> str:
    """Validates a stable identifier (claim id, framework id, control id)
    is a kebab-case slug, raising with `label` naming which kind it is. One
    rule shared by every id that ends up in a `<framework>:<control>` style
    reference, where a stray underscore or capital would silently fail to
    resolve."""
    if not _KEBAB_SLUG_RE.match(value):
        raise ValueError(f"{label} {value!r} is not a kebab-case slug")
    return value


def first_duplicate(values: Iterable[str]) -> str | None:
    """Returns the first value seen twice in an iterable of strings, or None
    if all are distinct. Shared by every uniqueness check that only needs to
    name one offender to raise a useful error."""
    seen: set[str] = set()
    for value in values:
        if value in seen:
            return value
        seen.add(value)
    return None


class FrameworkType(StrEnum):
    """What kind of thing a compliance framework actually is -- a binding
    regulation, a sector-specific regulation, a certification a business
    chooses to pursue, or an attestation a third party performs."""

    REGULATION = "regulation"
    SECTOR_REGULATION = "sector-regulation"
    CERTIFICATION = "certification"
    ATTESTATION = "attestation"


class FrameworkControl(BaseModel):
    """One control within a compliance framework -- e.g. "Art. 14, Human
    oversight" inside the EU AI Act. `id` is the stable slug a `control_ref`
    points at; `ref` and `title` are what a human reads."""

    id: str
    ref: str
    title: str

    @field_validator("id")
    @classmethod
    def _id_is_kebab_slug(cls, value: str) -> str:
        return require_kebab_slug(value, "control id")


class Framework(BaseModel):
    """One compliance framework and the controls a claim or an attested
    action can be evidence for. Control ids must be unique within the
    framework -- a ref resolves by (framework id, control id), and a
    duplicate would make that lookup ambiguous."""

    id: str
    name: str
    type: FrameworkType
    controls: list[FrameworkControl]

    @field_validator("id")
    @classmethod
    def _id_is_kebab_slug(cls, value: str) -> str:
        return require_kebab_slug(value, "framework id")

    @model_validator(mode="after")
    def _control_ids_unique(self) -> Framework:
        dupe = first_duplicate(control.id for control in self.controls)
        if dupe is not None:
            raise ValueError(f"framework {self.id!r} has duplicate control id {dupe!r}")
        return self


class FrameworkRegistry(BaseModel):
    """The whole compliance-framework catalogue: a disclaimer (this is a
    directional mapping, not legal advice) plus the frameworks. Framework
    ids must be unique across entries for the same reason control ids must
    be unique within a framework -- a ref resolves against this registry by
    framework id."""

    disclaimer: str
    entries: list[Framework]

    @model_validator(mode="after")
    def _framework_ids_unique(self) -> FrameworkRegistry:
        dupe = first_duplicate(framework.id for framework in self.entries)
        if dupe is not None:
            raise ValueError(f"duplicate framework id: {dupe!r}")
        return self


def load_frameworks(path: Path | str | None = None) -> FrameworkRegistry:
    """Parses and validates the framework catalogue. With no argument it
    reads the committed package catalogue `compliance/frameworks.yaml` --
    the one source both the story pipeline and the evidence-pack tagger
    resolve refs against. Raises `ValueError` on any malformed entry."""
    catalogue_path = Path(path) if path is not None else _DEFAULT_CATALOGUE
    raw: Any = yaml.safe_load(catalogue_path.read_text())
    return FrameworkRegistry.model_validate(raw)
