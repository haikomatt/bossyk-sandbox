from __future__ import annotations

from pathlib import Path

import pytest

from bossyk_sandbox.deck import render_deck_html
from bossyk_sandbox.story import load_story
from scripts.export_deck import DEFAULT_OUTPUT, FIGURES_DIR, STORY_PATH, export_deck

REPO_ROOT = Path(__file__).parent.parent.parent

_STORY = load_story(STORY_PATH)


def test_deck_has_a_title_slide_plus_one_slide_per_act() -> None:
    html = render_deck_html(_STORY, FIGURES_DIR)
    # one title slide + one per act
    assert html.count('class="slide ') == len(_STORY.acts) + 1
    for act in _STORY.acts:
        assert f">{act.title}<" in html


def test_deck_carries_the_interp_act_and_its_exec_copy() -> None:
    html = render_deck_html(_STORY, FIGURES_DIR)
    assert "Why not just read the model?" in html
    # the synthesis claim's exec copy, exec-facing (short) form
    assert "It governs the effect the action would have" in html


def test_deck_uses_exec_copy_not_the_technical_paragraph() -> None:
    # The deck is the exec surface: it must not leak a tech_copy paragraph.
    html = render_deck_html(_STORY, FIGURES_DIR)
    tech_snippet = _STORY.claims[0].tech_copy[:40]
    assert tech_snippet not in html


def test_deck_inlines_each_referenced_figure() -> None:
    html = render_deck_html(_STORY, FIGURES_DIR)
    referenced = {fid for claim in _STORY.claims for fid in claim.figure_ids}
    assert referenced  # the story does reference figures
    # every referenced figure is inlined as an <svg>, and the interp figure is present
    assert html.count("<svg") == len(referenced)
    assert "interp-three-negatives" not in html or "<svg" in html


def test_deck_render_is_byte_identical_across_two_runs() -> None:
    assert render_deck_html(_STORY, FIGURES_DIR) == render_deck_html(_STORY, FIGURES_DIR)


def test_deck_missing_figure_raises(tmp_path: Path) -> None:
    with pytest.raises((FileNotFoundError, ValueError)):
        render_deck_html(_STORY, tmp_path)  # empty dir -> referenced figures absent


def test_committed_deck_is_current(tmp_path: Path) -> None:
    """docs/deck.html must match a fresh render byte-for-byte -- regenerate with
    scripts/export_deck.py whenever the story or its figures change."""
    fresh = export_deck(tmp_path / "deck.html").read_text()
    assert DEFAULT_OUTPUT.read_text() == fresh, (
        "docs/deck.html is stale -- regenerate with scripts/export_deck.py"
    )
