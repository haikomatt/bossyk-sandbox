#!/usr/bin/env python
"""Lints `story/story.yaml` -- the single source of narrative truth for the
demo/deck pipeline -- against the committed benchmark artifacts it cites.

Loads the story (schema + cross-referential checks: unique claim ids,
every claim's act defined, every artifact_ref/figure_id exists on disk --
see `bossyk_sandbox.story.load_story`), then checks every claim's
`numeric_checks` against the artifact JSONs they name (see
`bossyk_sandbox.story.lint_story`). Deterministic: no network, no keys.

Usage:
    uv run python scripts/story_lint.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from bossyk_sandbox.story import lint_story, load_story

REPO_ROOT = Path(__file__).parent.parent
STORY_PATH = REPO_ROOT / "story" / "story.yaml"


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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
