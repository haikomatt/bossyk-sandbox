#!/usr/bin/env python
"""Regenerates the committed exec deck (`docs/deck.html`) from
`story/story.yaml` and the committed `docs/figures/*.svg`. Deterministic: no
network, no wall-clock -- re-running reproduces the file byte-for-byte (see
`tests/unit/test_deck.py`). Regenerate + commit whenever the story or its
figures change.

Usage:
    uv run python scripts/export_deck.py [--output PATH]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from bossyk_sandbox.deck import render_deck_html
from bossyk_sandbox.story import load_story

REPO_ROOT = Path(__file__).parent.parent
STORY_PATH = REPO_ROOT / "story" / "story.yaml"
FIGURES_DIR = REPO_ROOT / "docs" / "figures"
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "deck.html"


def export_deck(output: Path) -> Path:
    story = load_story(STORY_PATH)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_deck_html(story, FIGURES_DIR))
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Path to write the deck HTML to (default: docs/deck.html).",
    )
    args = parser.parse_args(argv)
    written = export_deck(Path(args.output))
    print(str(written))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
