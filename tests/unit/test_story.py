from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from bossyk_sandbox.story import EvidenceGrade, load_story

REPO_ROOT = Path(__file__).parent.parent.parent


def _act(act: int, title: str = "Title", question: str = "Question?") -> dict[str, Any]:
    return {"act": act, "title": title, "question": question}


def _claim(
    *,
    id: str = "some-claim",
    act: int = 1,
    exec_copy: str = "A plain sentence.",
    tech_copy: str = "A short paragraph with the mechanism and the caveat.",
    verdict: str | None = None,
    evidence_grade: str = "live-measurement",
    artifact_refs: list[str] | None = None,
    figure_ids: list[str] | None = None,
    numeric_checks: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "id": id,
        "act": act,
        "exec_copy": exec_copy,
        "tech_copy": tech_copy,
        "verdict": verdict,
        "evidence_grade": evidence_grade,
        "artifact_refs": artifact_refs or [],
        "figure_ids": figure_ids or [],
        "numeric_checks": numeric_checks or [],
    }


def _six_acts() -> list[dict[str, Any]]:
    return [_act(n) for n in range(1, 7)]


def _write_story(tmp_path: Path, story: dict[str, Any]) -> Path:
    story_dir = tmp_path / "story"
    story_dir.mkdir(parents=True, exist_ok=True)
    path = story_dir / "story.yaml"
    path.write_text(yaml.safe_dump(story))
    return path


def _write_artifact(tmp_path: Path, relative_path: str, payload: dict[str, Any]) -> Path:
    path = tmp_path / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return path


def _write_figure(tmp_path: Path, figure_id: str) -> Path:
    figures_dir = tmp_path / "docs" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    path = figures_dir / f"{figure_id}.svg"
    path.write_text("<svg></svg>")
    return path


def test_valid_story_loads(tmp_path: Path) -> None:
    _write_artifact(tmp_path, "docs/bench_output/fake.json", {"rate": 0.5})
    _write_figure(tmp_path, "some-figure")
    story_dict = {
        "acts": _six_acts(),
        "claims": [
            _claim(
                id="claim-one",
                act=1,
                artifact_refs=["docs/bench_output/fake.json"],
                figure_ids=["some-figure"],
                numeric_checks=[
                    {
                        "artifact": "docs/bench_output/fake.json",
                        "json_path": "rate",
                        "expected": 0.5,
                    }
                ],
            ),
            _claim(id="claim-two", act=2, verdict="pending", evidence_grade="open"),
        ],
    }
    path = _write_story(tmp_path, story_dict)

    story = load_story(path)

    assert len(story.acts) == 6
    assert {a.act for a in story.acts} == {1, 2, 3, 4, 5, 6}
    assert len(story.claims) == 2
    assert story.claims[0].id == "claim-one"
    assert story.claims[0].evidence_grade == EvidenceGrade.LIVE_MEASUREMENT
    assert story.claims[1].verdict == "pending"


def test_duplicate_claim_id_raises_naming_the_id(tmp_path: Path) -> None:
    story_dict = {
        "acts": _six_acts(),
        "claims": [
            _claim(id="dup-claim", act=1),
            _claim(id="dup-claim", act=2),
        ],
    }
    path = _write_story(tmp_path, story_dict)

    with pytest.raises(ValueError, match="dup-claim"):
        load_story(path)


def test_claim_referencing_unknown_act_raises_naming_the_offender(tmp_path: Path) -> None:
    # 6 act entries, but two both numbered 5 -- no act is numbered 6, so a
    # claim referencing act=6 has nothing to bind to.
    acts = [_act(1), _act(2), _act(3), _act(4), _act(5), _act(5)]
    story_dict = {
        "acts": acts,
        "claims": [_claim(id="orphan-claim", act=6)],
    }
    path = _write_story(tmp_path, story_dict)

    with pytest.raises(ValueError, match="orphan-claim"):
        load_story(path)


def test_missing_artifact_ref_raises_naming_the_path(tmp_path: Path) -> None:
    story_dict = {
        "acts": _six_acts(),
        "claims": [
            _claim(
                id="missing-artifact-claim",
                act=1,
                artifact_refs=["docs/bench_output/does_not_exist.json"],
            )
        ],
    }
    path = _write_story(tmp_path, story_dict)

    with pytest.raises(ValueError, match="does_not_exist.json"):
        load_story(path)


def test_missing_figure_id_raises_naming_the_offender(tmp_path: Path) -> None:
    story_dict = {
        "acts": _six_acts(),
        "claims": [
            _claim(
                id="missing-figure-claim",
                act=1,
                figure_ids=["nonexistent-figure"],
            )
        ],
    }
    path = _write_story(tmp_path, story_dict)

    with pytest.raises(ValueError, match="nonexistent-figure"):
        load_story(path)


def test_claim_id_must_be_kebab_slug(tmp_path: Path) -> None:
    story_dict = {
        "acts": _six_acts(),
        "claims": [_claim(id="Not_A_Slug", act=1)],
    }
    path = _write_story(tmp_path, story_dict)

    with pytest.raises(ValueError):
        load_story(path)


def test_claim_act_out_of_range_raises(tmp_path: Path) -> None:
    story_dict = {
        "acts": _six_acts(),
        "claims": [_claim(id="bad-act-claim", act=7)],
    }
    path = _write_story(tmp_path, story_dict)

    with pytest.raises(ValueError):
        load_story(path)


def test_acts_must_be_exactly_six(tmp_path: Path) -> None:
    story_dict = {
        "acts": [_act(n) for n in range(1, 6)],  # only 5
        "claims": [_claim(id="only-claim", act=1)],
    }
    path = _write_story(tmp_path, story_dict)

    with pytest.raises(ValueError, match="6"):
        load_story(path)


def test_claims_must_be_non_empty(tmp_path: Path) -> None:
    story_dict = {"acts": _six_acts(), "claims": []}
    path = _write_story(tmp_path, story_dict)

    with pytest.raises(ValueError):
        load_story(path)


# --- lint_story -------------------------------------------------------


def test_lint_catches_a_wrong_number(tmp_path: Path) -> None:
    from bossyk_sandbox.story import lint_story

    _write_artifact(tmp_path, "docs/bench_output/fake.json", {"nested": {"rate": 0.194}})
    story_dict = {
        "acts": _six_acts(),
        "claims": [
            _claim(
                id="wrong-number-claim",
                act=1,
                numeric_checks=[
                    {
                        "artifact": "docs/bench_output/fake.json",
                        "json_path": "nested.rate",
                        "expected": 0.5,
                    }
                ],
            )
        ],
    }
    path = _write_story(tmp_path, story_dict)
    story = load_story(path)

    failures = lint_story(story, tmp_path)

    assert len(failures) == 1
    assert "wrong-number-claim" in failures[0]


def test_lint_passes_a_right_number(tmp_path: Path) -> None:
    from bossyk_sandbox.story import lint_story

    _write_artifact(tmp_path, "docs/bench_output/fake.json", {"nested": {"rate": 0.194}})
    story_dict = {
        "acts": _six_acts(),
        "claims": [
            _claim(
                id="right-number-claim",
                act=1,
                numeric_checks=[
                    {
                        "artifact": "docs/bench_output/fake.json",
                        "json_path": "nested.rate",
                        "expected": 0.194,
                    }
                ],
            )
        ],
    }
    path = _write_story(tmp_path, story_dict)
    story = load_story(path)

    failures = lint_story(story, tmp_path)

    assert failures == []


def test_lint_json_path_supports_list_indexing(tmp_path: Path) -> None:
    from bossyk_sandbox.story import lint_story

    _write_artifact(
        tmp_path,
        "docs/bench_output/fake.json",
        {"steps": [{"gate_verdict": "block"}, {"rate": 0.4166666666666667}]},
    )
    story_dict = {
        "acts": _six_acts(),
        "claims": [
            _claim(
                id="indexed-claim",
                act=1,
                numeric_checks=[
                    {
                        "artifact": "docs/bench_output/fake.json",
                        "json_path": "steps[1].rate",
                        "expected": 0.4166666666666667,
                    }
                ],
            )
        ],
    }
    path = _write_story(tmp_path, story_dict)
    story = load_story(path)

    failures = lint_story(story, tmp_path)

    assert failures == []


def test_lint_string_expected_mismatch_is_reported(tmp_path: Path) -> None:
    from bossyk_sandbox.story import lint_story

    _write_artifact(tmp_path, "docs/bench_output/fake.json", {"canonical_strength": "leaky"})
    story_dict = {
        "acts": _six_acts(),
        "claims": [
            _claim(
                id="string-mismatch-claim",
                act=1,
                numeric_checks=[
                    {
                        "artifact": "docs/bench_output/fake.json",
                        "json_path": "canonical_strength",
                        "expected": "strict",
                    }
                ],
            )
        ],
    }
    path = _write_story(tmp_path, story_dict)
    story = load_story(path)

    failures = lint_story(story, tmp_path)

    assert len(failures) == 1
    assert "string-mismatch-claim" in failures[0]


# --- CLI acceptance check (written last, after story/story.yaml + figures exist) --


def test_cli_exits_zero_on_the_real_committed_story(tmp_path: Path) -> None:
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "scripts/story_lint.py"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
