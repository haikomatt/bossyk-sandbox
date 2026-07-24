#!/usr/bin/env python
"""Lints `story/story.yaml` -- the single source of narrative truth for the
demo/deck pipeline -- against the committed benchmark artifacts it cites,
then prints the compliance-framework coverage report.

Loads the story (schema + cross-referential checks: unique claim ids,
every claim's act defined, every artifact_ref/figure_id/control_ref
resolves -- see `bossyk_sandbox.story.load_story`), then checks every
claim's `numeric_checks` against the artifact JSONs they name (see
`bossyk_sandbox.story.lint_story`). Finally reports, per framework, which
controls the story's claims actually back and which are still uncovered
(see `bossyk_sandbox.story.coverage_by_framework`). Deterministic: no
network, no keys.

Usage:
    uv run python scripts/story_lint.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from bossyk_sandbox.story import Story, coverage_by_framework, lint_story, load_story

REPO_ROOT = Path(__file__).parent.parent
STORY_PATH = REPO_ROOT / "story" / "story.yaml"


def _print_coverage_report(story: Story) -> None:
    """Prints per-framework control coverage. A control is 'covered' only
    if a claim graded above `open` backs it -- an open claim is a promise,
    not evidence -- so uncovered controls are named explicitly rather than
    left to look satisfied. Nothing to print if the story declares no
    frameworks block."""
    coverage = coverage_by_framework(story)
    if not coverage:
        return

    print("\ncompliance coverage (control backed by a claim graded above open):")
    for framework in coverage:
        covered = [control for control in framework.controls if control.is_covered]
        print(f"  {framework.framework.name}: {len(covered)}/{len(framework.controls)} controls")
        for control in framework.controls:
            mark = "OK " if control.is_covered else "-- "
            backing = ", ".join(f"{b.claim_id} ({b.evidence_grade})" for b in control.backing)
            detail = backing if backing else "no backing claim"
            print(f"    [{mark}] {control.ref} {control.title}: {detail}")
        uncovered = framework.uncovered
        if uncovered:
            print(f"    uncovered: {', '.join(control.control_id for control in uncovered)}")


def main(argv: list[str] | None = None) -> int:
    del argv  # no CLI args yet -- always lints the one committed story
    story = load_story(STORY_PATH)
    failures = lint_story(story, REPO_ROOT)

    if failures:
        for failure in failures:
            print(failure, file=sys.stderr)
        print(f"story_lint: {len(failures)} failure(s)", file=sys.stderr)
        return 1

    print(
        f"story_lint: OK -- {len(story.claims)} claim(s) across {len(story.acts)} acts, "
        "all numeric checks match"
    )
    _print_coverage_report(story)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
