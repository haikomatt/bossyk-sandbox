"""F0 story schema + loader + lint: `story/story.yaml` is the single source
of narrative truth for the demo/deck pipeline. Every quantitative claim it
makes must be checked against the committed benchmark artifacts (this
module's `lint_story`) so a slide deck (or the future demo UI) can never
disagree with the data those artifacts hold.

Reuses nothing from elsewhere in this repo -- this is the first module of
its kind (schema + lint over YAML/JSON), not a duplicate of any existing
scoring/merge module."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

_KEBAB_SLUG_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
# A json_path segment is a (possibly empty) dict key followed by zero or
# more `[int]` index chains, e.g. "domains", "steps[3]", "steps[3][0]".
_PATH_SEGMENT_RE = re.compile(r"^([^\[\]]*)((?:\[\d+\])*)$")
_INDEX_RE = re.compile(r"\[(\d+)\]")


class EvidenceGrade(StrEnum):
    """How a claim's numbers were produced -- the honesty axis carried
    over from the corrected phase5 synthesis: a claim must say whether it
    rests on a live measurement, a scripted proxy, a modeled
    counterfactual, a deterministic recompute of persisted judge verdicts,
    or is still open."""

    LIVE_MEASUREMENT = "live-measurement"
    SCRIPTED_PROXY = "scripted-proxy"
    MODELED_COUNTERFACTUAL = "modeled-counterfactual"
    DETERMINISTIC_RECOMPUTE = "deterministic-recompute"
    OPEN = "open"


class FrameworkType(StrEnum):
    """What kind of thing a compliance framework actually is -- a binding
    regulation, a sector-specific regulation, a certification a business
    chooses to pursue, or an attestation a third party performs. Kept
    separate from `EvidenceGrade`: this describes the framework itself,
    not how well any given claim proves compliance with it."""

    REGULATION = "regulation"
    SECTOR_REGULATION = "sector-regulation"
    CERTIFICATION = "certification"
    ATTESTATION = "attestation"


class FrameworkControl(BaseModel):
    """One control within a compliance framework -- e.g. "Art. 14, Human
    oversight" inside the EU AI Act. `id` is the stable slug a claim's
    `control_refs` points at; `ref` and `title` are what a human reads."""

    id: str
    ref: str
    title: str

    @field_validator("id")
    @classmethod
    def _id_is_kebab_slug(cls, value: str) -> str:
        if not _KEBAB_SLUG_RE.match(value):
            raise ValueError(f"control id {value!r} is not a kebab-case slug")
        return value


class Framework(BaseModel):
    """One compliance framework and the controls within it a claim can be
    evidence for. Control ids must be unique within the framework -- a
    story claim resolves a control_ref by (framework id, control id), and
    a duplicate would make that lookup ambiguous."""

    id: str
    name: str
    type: FrameworkType
    controls: list[FrameworkControl]

    @field_validator("id")
    @classmethod
    def _id_is_kebab_slug(cls, value: str) -> str:
        if not _KEBAB_SLUG_RE.match(value):
            raise ValueError(f"framework id {value!r} is not a kebab-case slug")
        return value

    @model_validator(mode="after")
    def _control_ids_unique(self) -> Framework:
        dupe = _first_duplicate(control.id for control in self.controls)
        if dupe is not None:
            raise ValueError(f"framework {self.id!r} has duplicate control id {dupe!r}")
        return self


class FrameworkRegistry(BaseModel):
    """The story's whole compliance-framework catalogue: a disclaimer
    (this is a directional mapping, not legal advice) plus the frameworks
    themselves. Framework ids must be unique across entries for the same
    reason control ids must be unique within a framework -- a control_ref
    resolves against this registry by framework id."""

    disclaimer: str
    entries: list[Framework]

    @model_validator(mode="after")
    def _framework_ids_unique(self) -> FrameworkRegistry:
        dupe = _first_duplicate(framework.id for framework in self.entries)
        if dupe is not None:
            raise ValueError(f"duplicate framework id: {dupe!r}")
        return self


class NumericCheck(BaseModel):
    """One number a claim asserts, checked by `lint_story` against a
    committed artifact JSON at `artifact` (a repo-relative path) by
    resolving `json_path` and comparing to `expected`."""

    artifact: str
    json_path: str
    expected: float | str


class StoryClaim(BaseModel):
    """One narrative claim: a plain-English sentence (`exec_copy`) backed
    by a short technical paragraph (`tech_copy`), graded by how it was
    produced (`evidence_grade`), checkable against real artifacts
    (`artifact_refs`, `numeric_checks`), and optionally tagged with which
    regulatory controls it counts as evidence for (`control_refs`) -- an
    orthogonal axis: `evidence_grade` says how good the number is,
    `control_refs` says which control it is evidence FOR."""

    id: str
    act: int
    exec_copy: str
    tech_copy: str
    verdict: str | None = None
    evidence_grade: EvidenceGrade
    artifact_refs: list[str] = Field(default_factory=list)
    figure_ids: list[str] = Field(default_factory=list)
    numeric_checks: list[NumericCheck] = Field(default_factory=list)
    control_refs: list[str] = Field(default_factory=list)

    @field_validator("id")
    @classmethod
    def _id_is_kebab_slug(cls, value: str) -> str:
        if not _KEBAB_SLUG_RE.match(value):
            raise ValueError(f"claim id {value!r} is not a kebab-case slug")
        return value

    @field_validator("act")
    @classmethod
    def _act_in_range(cls, value: int) -> int:
        if not 1 <= value <= 6:
            raise ValueError(f"claim act {value} must be between 1 and 6")
        return value


class StoryAct(BaseModel):
    """One of the demo's 6 acts: a title + the question it answers."""

    act: int
    title: str
    question: str

    @field_validator("act")
    @classmethod
    def _act_in_range(cls, value: int) -> int:
        if not 1 <= value <= 6:
            raise ValueError(f"act {value} must be between 1 and 6")
        return value


class Story(BaseModel):
    """The whole narrative: exactly 6 acts, at least one claim. Cross-
    referential checks (claim ids unique, every claim's act actually
    defined, artifact_refs/figure_ids exist on disk) are load_story's job,
    not this model's -- they need a repo root and the filesystem, which a
    pydantic validator has no clean access to."""

    acts: list[StoryAct]
    claims: list[StoryClaim]
    frameworks: FrameworkRegistry | None = None

    @model_validator(mode="after")
    def _validate_shape(self) -> Story:
        if len(self.acts) != 6:
            raise ValueError(f"story must define exactly 6 acts, got {len(self.acts)}")
        if not self.claims:
            raise ValueError("story must define at least one claim")
        return self


class ClaimGrade(BaseModel):
    """One claim backing a control, paired with the evidence_grade it
    carries -- `coverage_by_framework` needs the grade alongside the
    identity so callers can apply the anti-coverage-theatre rule (a claim
    graded `open` is a promise, not evidence) without a second lookup."""

    claim_id: str
    evidence_grade: EvidenceGrade


class ControlCoverage(BaseModel):
    """One framework control plus every claim in the story that names it
    via `control_refs`, in story-claim order."""

    framework_id: str
    control_id: str
    ref: str
    title: str
    backing: list[ClaimGrade]

    @property
    def is_covered(self) -> bool:
        """True only if at least one backing claim's evidence_grade is
        something other than `open` -- the anti-coverage-theatre rule.
        A control cited only by open (still-to-be-measured) claims has no
        actual evidence behind it yet, so it must not read as covered."""
        return any(backing.evidence_grade != EvidenceGrade.OPEN for backing in self.backing)


class FrameworkCoverage(BaseModel):
    """Coverage of a single framework's controls by the story's claims."""

    framework: Framework
    controls: list[ControlCoverage]

    @property
    def uncovered(self) -> list[ControlCoverage]:
        """The controls for which `is_covered` is False, in the same
        (declaration) order as `controls`."""
        return [control for control in self.controls if not control.is_covered]


def _first_duplicate(values: Iterable[str]) -> str | None:
    """Returns the first value seen twice in an iterable of strings, or
    None if all values are distinct. Shared by every uniqueness check in
    this module (claim ids, framework ids, control ids within a
    framework) that only needs to name one offender to raise a useful
    error."""
    seen: set[str] = set()
    for value in values:
        if value in seen:
            return value
        seen.add(value)
    return None


def _find_duplicate_ids(claims: list[StoryClaim]) -> list[str]:
    seen: set[str] = set()
    dupes: list[str] = []
    for claim in claims:
        if claim.id in seen and claim.id not in dupes:
            dupes.append(claim.id)
        seen.add(claim.id)
    return dupes


def load_story(path: Path | str) -> Story:
    """Parses + pydantic-validates a story YAML file, then runs the
    cross-referential checks that need the filesystem:

    - claim ids are unique across the story;
    - every claim's `act` matches an `act` value actually present in
      `acts` (note: `Story` only requires exactly 6 act entries, each
      individually in range 1-6 -- it does not require them to be a
      1..6 bijection, so this check is not vacuous);
    - every `artifact_refs` path exists on disk, relative to the repo
      root;
    - every `figure_ids` entry has a corresponding
      `docs/figures/<id>.svg`.
    - every `control_refs` entry parses as `<framework-id>:<control-id>`
      and resolves to a defined framework and control in `frameworks`
      (a claim may carry no `control_refs` when `frameworks` is absent,
      but not any).

    Assumes the standard layout `<repo_root>/story/<name>.yaml` --
    the repo root used to resolve artifact_refs/figure_ids is
    `path.resolve().parent.parent`.

    Raises `ValueError` (pydantic's `ValidationError` is a `ValueError`
    subclass) naming the offending claim/path on any failure.
    """
    path = Path(path)
    raw: Any = yaml.safe_load(path.read_text())
    story = Story.model_validate(raw)

    repo_root = path.resolve().parent.parent

    dupes = _find_duplicate_ids(story.claims)
    if dupes:
        raise ValueError(f"duplicate claim id(s): {', '.join(sorted(dupes))}")

    defined_acts = {act.act for act in story.acts}
    for claim in story.claims:
        if claim.act not in defined_acts:
            raise ValueError(
                f"claim {claim.id!r} references act {claim.act}, which is not one of the "
                f"defined acts {sorted(defined_acts)}"
            )

        for ref in claim.artifact_refs:
            ref_path = repo_root / ref
            if not ref_path.exists():
                raise ValueError(
                    f"claim {claim.id!r} references artifact_ref {ref!r}, which does not "
                    f"exist at {ref_path}"
                )

        for figure_id in claim.figure_ids:
            figure_path = repo_root / "docs" / "figures" / f"{figure_id}.svg"
            if not figure_path.exists():
                raise ValueError(
                    f"claim {claim.id!r} references figure_id {figure_id!r}, which has no "
                    f"corresponding file at {figure_path}"
                )

        if claim.control_refs:
            if story.frameworks is None:
                raise ValueError(
                    f"claim {claim.id!r} references control_refs {claim.control_refs!r}, "
                    f"but the story defines no frameworks block to resolve them against"
                )
            frameworks = story.frameworks

            seen_control_refs: set[str] = set()
            for ref in claim.control_refs:
                if ref in seen_control_refs:
                    raise ValueError(
                        f"claim {claim.id!r} references control_ref {ref!r} more than once"
                    )
                seen_control_refs.add(ref)

                parts = ref.split(":")
                if len(parts) != 2 or not parts[0] or not parts[1]:
                    raise ValueError(
                        f"claim {claim.id!r} references control_ref {ref!r}, which is not "
                        f"in '<framework-id>:<control-id>' form"
                    )
                framework_id, control_id = parts

                framework = next((f for f in frameworks.entries if f.id == framework_id), None)
                if framework is None:
                    raise ValueError(
                        f"claim {claim.id!r} references control_ref {ref!r}, whose "
                        f"framework {framework_id!r} is not defined"
                    )

                control = next((c for c in framework.controls if c.id == control_id), None)
                if control is None:
                    raise ValueError(
                        f"claim {claim.id!r} references control_ref {ref!r}, whose "
                        f"control {control_id!r} is not defined in framework "
                        f"{framework_id!r}"
                    )

    return story


def _resolve_json_path(data: Any, json_path: str) -> Any:
    """Resolves a dotted json_path (dict keys, each optionally followed by
    one or more `[int]` index chains) against a parsed JSON structure --
    e.g. `domains.airline.h1_overall.rate` or `steps[3].gate_verdict`."""
    node = data
    for segment in json_path.split("."):
        match = _PATH_SEGMENT_RE.match(segment)
        if match is None:
            raise ValueError(f"malformed json_path segment {segment!r} in {json_path!r}")
        key, index_chain = match.group(1), match.group(2)
        if key:
            if not isinstance(node, dict) or key not in node:
                raise ValueError(f"json_path {json_path!r}: key {key!r} not found")
            node = node[key]
        for index_str in _INDEX_RE.findall(index_chain):
            index = int(index_str)
            if not isinstance(node, list) or index >= len(node):
                raise ValueError(f"json_path {json_path!r}: index {index} out of range")
            node = node[index]
    return node


def lint_story(story: Story, repo_root: Path | str) -> list[str]:
    """Checks every claim's `numeric_checks` against the committed
    artifact JSONs named by `artifact` (repo-relative to `repo_root`).
    Returns a list of human-readable failure strings naming the claim,
    artifact, and json_path (empty list = clean).

    Never raises for a data problem (missing artifact, malformed
    json_path, type/value mismatch) -- those are collected as failure
    strings so one bad claim doesn't abort the rest of the lint run.
    Floats are compared with `abs(actual - expected) < 1e-9`; strings with
    exact equality."""
    repo_root = Path(repo_root)
    failures: list[str] = []
    artifact_cache: dict[str, Any] = {}

    for claim in story.claims:
        for check in claim.numeric_checks:
            if check.artifact in artifact_cache:
                data = artifact_cache[check.artifact]
            else:
                artifact_path = repo_root / check.artifact
                if not artifact_path.exists():
                    failures.append(
                        f"{claim.id}: artifact {check.artifact!r} does not exist at {artifact_path}"
                    )
                    continue
                data = json.loads(artifact_path.read_text())
                artifact_cache[check.artifact] = data

            try:
                actual = _resolve_json_path(data, check.json_path)
            except ValueError as exc:
                failures.append(f"{claim.id}: {exc}")
                continue

            expected = check.expected
            if isinstance(expected, str):
                if actual != expected:
                    failures.append(
                        f"{claim.id}: {check.artifact}#{check.json_path} = {actual!r}, "
                        f"expected {expected!r}"
                    )
            elif not isinstance(actual, int | float) or isinstance(actual, bool):
                failures.append(
                    f"{claim.id}: {check.artifact}#{check.json_path} = {actual!r} is not "
                    f"numeric, expected {expected}"
                )
            elif abs(float(actual) - expected) >= 1e-9:
                failures.append(
                    f"{claim.id}: {check.artifact}#{check.json_path} = {actual}, "
                    f"expected {expected}"
                )

    return failures


def coverage_by_framework(story: Story) -> list[FrameworkCoverage]:
    """Maps every framework control to the claims that cite it via
    `control_refs`, so a slide deck (or the future demo UI) can show which
    regulatory controls are actually backed by evidence rather than
    merely promised. `load_story` has already guaranteed every
    `control_ref` resolves, so this is a pure re-grouping -- no ValueError
    path here.

    Frameworks and controls are returned in declaration order and backing
    claims in story-claim order, because coverage_by_framework's whole
    purpose is being read by a human (via a deck or a UI), and dict/set
    iteration order must never leak into what a human reads.

    Returns `[]` when the story defines no `frameworks` block -- there is
    nothing to report coverage against.
    """
    if story.frameworks is None:
        return []

    coverage: list[FrameworkCoverage] = []
    for framework in story.frameworks.entries:
        controls: list[ControlCoverage] = []
        for control in framework.controls:
            control_ref = f"{framework.id}:{control.id}"
            backing = [
                ClaimGrade(claim_id=claim.id, evidence_grade=claim.evidence_grade)
                for claim in story.claims
                if control_ref in claim.control_refs
            ]
            controls.append(
                ControlCoverage(
                    framework_id=framework.id,
                    control_id=control.id,
                    ref=control.ref,
                    title=control.title,
                    backing=backing,
                )
            )
        coverage.append(FrameworkCoverage(framework=framework, controls=controls))

    return coverage
