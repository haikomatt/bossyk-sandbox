"""Regression: `preset_id` must not escape the presets directory.

Found 2026-09-16 by an independent threat-modelling run over this repo, then
confirmed by reading `console/replay.py`: `load_replay_preset` interpolated
`preset_id` straight into a path with no validation, so a `../` prefix escaped
`story/replay/`. `POST /session/replay` takes `preset_id` from the query string
with no authentication, making it remotely reachable.

The sibling artifact endpoint already had a tested guard for this exact class
(`tests/unit/test_artifacts_api.py::test_path_traversal_is_rejected`); this path
did not. The fix extends that guard rather than adding a second one.

NOTE ON TEST DESIGN: asserting `FileNotFoundError` for an arbitrary traversing
id is NOT a valid test of this bug. It passes trivially whenever the traversal
target happens not to exist, which proves nothing. The test must plant a
readable, correctly-shaped file OUTSIDE the presets directory and show it is
unreachable.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bossyk_sandbox.console import replay
from bossyk_sandbox.console.replay import load_replay_preset

_VALID_PRESET = {
    "preset_id": "planted",
    "title": "planted",
    "domain": "retail",
    "source_artifact": "none",
    "story_claim": "none",
    "turns": [
        {
            "tool_name": "get_order_details",
            "arguments": {},
            "label": "lookup",
            "role": "assistant",
        }
    ],
    "expected": {"verdicts": [], "harm_prevented": 0, "harm_delta": 0},
}


@pytest.fixture
def planted_outside(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """Point the presets dir at tmp_path/story/replay and plant a valid preset
    two levels up, i.e. outside it. Returns the traversing preset_id."""
    presets = tmp_path / "story" / "replay"
    presets.mkdir(parents=True)
    (tmp_path / "secret.json").write_text(json.dumps(_VALID_PRESET))
    monkeypatch.setattr(replay, "_PRESETS_DIR", presets)
    return "../../secret"


def test_traversing_preset_id_cannot_read_a_file_outside_the_presets_dir(
    planted_outside: str,
) -> None:
    """The planted file is valid and readable, so this fails ONLY because the
    traversal is blocked, not because the target is missing."""
    with pytest.raises(FileNotFoundError):
        load_replay_preset(planted_outside)


def test_the_planted_file_really_is_loadable_when_not_traversing(
    planted_outside: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Control for the test above: with the presets dir pointed AT the planted
    file's directory, the same preset loads. Without this, a bug in the fixture
    would make the traversal test pass vacuously."""
    monkeypatch.setattr(replay, "_PRESETS_DIR", tmp_path)
    assert load_replay_preset("secret").preset_id == "planted"
