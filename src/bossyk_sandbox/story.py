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
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from bossyk_sandbox.compliance.frameworks import (
    Framework,
    FrameworkControl,
    FrameworkRegistry,
    FrameworkType,
    first_duplicate,
    load_frameworks,
    require_kebab_slug,
)

# The framework catalogue models used to live here; they now belong to the
# story-independent `compliance` layer and are re-exported so existing
# `from bossyk_sandbox.story import Framework...` imports keep working.
__all__ = [
    "EvidenceGrade",
    "Framework",
    "FrameworkControl",
    "FrameworkRegistry",
    "FrameworkType",
    "Story",
    "StoryAct",
    "StoryClaim",
    "coverage_by_framework",
    "lint_story",
    "load_story",
]

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
        return require_kebab_slug(value, "claim id")

    @field_validator("act")
    @classmethod
    def _act_at_least_one(cls, value: int) -> int:
        if value < 1:
            raise ValueError(f"claim act {value} must be >= 1")
        return value


class StoryAct(BaseModel):
    """One of the demo's acts: a title + the question it answers. The act
    count is not fixed -- the story just requires a contiguous 1..N run
    (see `Story`), so a new act extends the narrative without a schema
    change."""

    act: int
    title: str
    question: str

    @field_validator("act")
    @classmethod
    def _act_at_least_one(cls, value: int) -> int:
        if value < 1:
            raise ValueError(f"act {value} must be >= 1")
        return value


class Story(BaseModel):
    """The whole narrative: a contiguous 1..N run of acts (N >= 1) and at
    least one claim. The act count is deliberately not pinned to a magic
    number so the narrative can grow an act without a schema change; what
    is enforced is that the act numbers are exactly 1..N with no gaps or
    duplicates, so every claim's `act` binds to a real act. Cross-
    referential checks (claim ids unique, every claim's act actually
    defined, artifact_refs/figure_ids exist on disk) are load_story's job,
    not this model's -- they need a repo root and the filesystem, which a
    pydantic validator has no clean access to."""

    acts: list[StoryAct]
    claims: list[StoryClaim]
    frameworks: FrameworkRegistry | None = None

    @model_validator(mode="after")
    def _validate_shape(self) -> Story:
        if not self.acts:
            raise ValueError("story must define at least one act")
        act_numbers = sorted(act.act for act in self.acts)
        if act_numbers != list(range(1, len(act_numbers) + 1)):
            raise ValueError(
                "story acts must be contiguously numbered 1..N with no gaps or "
                f"duplicates, got {act_numbers}"
            )
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


def load_story(path: Path | str, *, frameworks_path: Path | str | None = None) -> Story:
    """Parses + pydantic-validates a story YAML file, then runs the
    cross-referential checks that need the filesystem:

    - claim ids are unique across the story;
    - every claim's `act` matches an `act` value actually present in
      `acts` (note: `Story` requires the acts to be a contiguous 1..N run,
      but a claim's `act` only has to be >= 1, so a claim can still name an
      act beyond N -- e.g. act 7 of a six-act story -- and this check is
      what catches it);
    - every `artifact_refs` path exists on disk, relative to the repo
      root;
    - every `figure_ids` entry has a corresponding
      `docs/figures/<id>.svg`;
    - every `control_refs` entry parses as `<framework-id>:<control-id>`
      and resolves to a defined framework and control in the catalogue.

    The compliance catalogue is story-independent: a story normally omits
    any inline `frameworks` block, and `load_story` attaches the shared
    catalogue (`compliance/frameworks.yaml`, or `frameworks_path` if given)
    so every story resolves its control_refs against the one source of
    truth. A story that does inline a `frameworks` block keeps it (used in
    tests); there is always a catalogue by the time refs are resolved.

    Assumes the standard layout `<repo_root>/story/<name>.yaml` --
    the repo root used to resolve artifact_refs/figure_ids is
    `path.resolve().parent.parent`.

    Raises `ValueError` (pydantic's `ValidationError` is a `ValueError`
    subclass) naming the offending claim/path on any failure.
    """
    path = Path(path)
    raw: Any = yaml.safe_load(path.read_text())
    story = Story.model_validate(raw)

    catalogue = (
        story.frameworks if story.frameworks is not None else load_frameworks(frameworks_path)
    )
    story.frameworks = catalogue

    repo_root = path.resolve().parent.parent

    dupe_claim_id = first_duplicate(claim.id for claim in story.claims)
    if dupe_claim_id is not None:
        raise ValueError(f"duplicate claim id: {dupe_claim_id!r}")

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

        _resolve_control_refs(claim, catalogue)

    return story


def _resolve_control_refs(claim: StoryClaim, frameworks: FrameworkRegistry) -> None:
    """Checks every `control_ref` on `claim` parses as
    `<framework-id>:<control-id>`, is not repeated, and resolves to a
    framework and control actually defined in the catalogue. Raises
    `ValueError` naming the claim first, then the offending ref. A claim
    with no `control_refs` is a no-op. `load_story` always attaches a
    catalogue before calling this, so there is always one to resolve
    against."""
    if not claim.control_refs:
        return

    seen: set[str] = set()
    for ref in claim.control_refs:
        if ref in seen:
            raise ValueError(f"claim {claim.id!r} references control_ref {ref!r} more than once")
        seen.add(ref)

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

        if not any(control.id == control_id for control in framework.controls):
            raise ValueError(
                f"claim {claim.id!r} references control_ref {ref!r}, whose "
                f"control {control_id!r} is not defined in framework {framework_id!r}"
            )


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
