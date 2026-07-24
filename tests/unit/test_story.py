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
    control_refs: list[str] | None = None,
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
        "control_refs": control_refs or [],
    }


def _control(
    id: str = "art-12",
    ref: str = "Art. 12",
    title: str = "Automatic event logging",
) -> dict[str, Any]:
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


def _frameworks(
    *,
    entries: list[dict[str, Any]] | None = None,
    disclaimer: str = "Directional mapping, not legal advice.",
) -> dict[str, Any]:
    return {
        "disclaimer": disclaimer,
        "entries": entries if entries is not None else [_framework()],
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


# --- compliance frameworks -------------------------------------------
#
# The second, orthogonal axis: evidence_grade says how good a number is,
# control_refs say which regulatory control it is evidence FOR. The two
# must never collapse into one another -- see the coverage tests below,
# where an `open`-graded claim deliberately does NOT cover its control.


def test_frameworks_block_loads_with_its_controls(tmp_path: Path) -> None:
    from bossyk_sandbox.story import FrameworkType

    story_dict = {
        "acts": _six_acts(),
        "frameworks": _frameworks(
            entries=[
                _framework(
                    id="eu-ai-act",
                    controls=[_control("art-12"), _control("art-14", "Art. 14", "Human oversight")],
                ),
                _framework(id="soc2", name="SOC 2", type="attestation"),
            ]
        ),
        "claims": [_claim(id="claim-one", act=1)],
    }
    path = _write_story(tmp_path, story_dict)

    story = load_story(path)

    assert story.frameworks is not None
    assert [f.id for f in story.frameworks.entries] == ["eu-ai-act", "soc2"]
    assert story.frameworks.entries[0].type == FrameworkType.REGULATION
    assert story.frameworks.entries[1].type == FrameworkType.ATTESTATION
    assert [c.id for c in story.frameworks.entries[0].controls] == ["art-12", "art-14"]
    assert story.frameworks.entries[0].controls[1].title == "Human oversight"
    assert "not legal advice" in story.frameworks.disclaimer


def test_claim_control_refs_resolve_against_the_frameworks_block(tmp_path: Path) -> None:
    story_dict = {
        "acts": _six_acts(),
        "frameworks": _frameworks(
            entries=[
                _framework(id="eu-ai-act", controls=[_control("art-12"), _control("art-14")]),
            ]
        ),
        "claims": [_claim(id="tagged-claim", act=1, control_refs=["eu-ai-act:art-14"])],
    }
    path = _write_story(tmp_path, story_dict)

    story = load_story(path)

    assert story.claims[0].control_refs == ["eu-ai-act:art-14"]


def test_control_ref_to_unknown_framework_raises_naming_the_claim_and_ref(tmp_path: Path) -> None:
    story_dict = {
        "acts": _six_acts(),
        "frameworks": _frameworks(),
        "claims": [_claim(id="bad-framework-claim", act=1, control_refs=["nist-ai-rmf:govern"])],
    }
    path = _write_story(tmp_path, story_dict)

    with pytest.raises(ValueError, match="bad-framework-claim.*nist-ai-rmf:govern"):
        load_story(path)


def test_control_ref_to_unknown_control_raises_naming_the_claim_and_ref(tmp_path: Path) -> None:
    story_dict = {
        "acts": _six_acts(),
        "frameworks": _frameworks(
            entries=[_framework(id="eu-ai-act", controls=[_control("art-12")])]
        ),
        "claims": [_claim(id="bad-control-claim", act=1, control_refs=["eu-ai-act:art-99"])],
    }
    path = _write_story(tmp_path, story_dict)

    with pytest.raises(ValueError, match="bad-control-claim.*eu-ai-act:art-99"):
        load_story(path)


def test_control_ref_without_a_framework_block_raises(tmp_path: Path) -> None:
    # A story may omit `frameworks` entirely (backward compatible), but then
    # no claim may carry a control_ref -- there is nothing to resolve against.
    story_dict = {
        "acts": _six_acts(),
        "claims": [_claim(id="unresolvable-claim", act=1, control_refs=["eu-ai-act:art-12"])],
    }
    path = _write_story(tmp_path, story_dict)

    with pytest.raises(ValueError, match="unresolvable-claim"):
        load_story(path)


def test_story_without_a_frameworks_block_still_loads(tmp_path: Path) -> None:
    story_dict = {
        "acts": _six_acts(),
        "claims": [_claim(id="untagged-claim", act=1)],
    }
    path = _write_story(tmp_path, story_dict)

    story = load_story(path)

    assert story.frameworks is None
    assert story.claims[0].control_refs == []


@pytest.mark.parametrize("malformed", ["eu-ai-act", "eu-ai-act:", ":art-12", "a:b:c"])
def test_malformed_control_ref_raises(tmp_path: Path, malformed: str) -> None:
    story_dict = {
        "acts": _six_acts(),
        "frameworks": _frameworks(),
        "claims": [_claim(id="malformed-ref-claim", act=1, control_refs=[malformed])],
    }
    path = _write_story(tmp_path, story_dict)

    with pytest.raises(ValueError):
        load_story(path)


def test_duplicate_control_refs_on_one_claim_raises(tmp_path: Path) -> None:
    story_dict = {
        "acts": _six_acts(),
        "frameworks": _frameworks(),
        "claims": [
            _claim(
                id="repeated-ref-claim",
                act=1,
                control_refs=["eu-ai-act:art-12", "eu-ai-act:art-12"],
            )
        ],
    }
    path = _write_story(tmp_path, story_dict)

    with pytest.raises(ValueError, match="eu-ai-act:art-12"):
        load_story(path)


def test_duplicate_framework_ids_raise(tmp_path: Path) -> None:
    story_dict = {
        "acts": _six_acts(),
        "frameworks": _frameworks(entries=[_framework(id="soc2"), _framework(id="soc2")]),
        "claims": [_claim(id="claim-one", act=1)],
    }
    path = _write_story(tmp_path, story_dict)

    with pytest.raises(ValueError, match="soc2"):
        load_story(path)


def test_duplicate_control_ids_within_a_framework_raise(tmp_path: Path) -> None:
    story_dict = {
        "acts": _six_acts(),
        "frameworks": _frameworks(
            entries=[_framework(id="eu-ai-act", controls=[_control("art-12"), _control("art-12")])]
        ),
        "claims": [_claim(id="claim-one", act=1)],
    }
    path = _write_story(tmp_path, story_dict)

    with pytest.raises(ValueError, match="art-12"):
        load_story(path)


def test_framework_and_control_ids_must_be_kebab_slugs(tmp_path: Path) -> None:
    story_dict = {
        "acts": _six_acts(),
        "frameworks": _frameworks(entries=[_framework(id="EU_AI_Act")]),
        "claims": [_claim(id="claim-one", act=1)],
    }
    path = _write_story(tmp_path, story_dict)

    with pytest.raises(ValueError):
        load_story(path)


def test_unknown_framework_type_raises(tmp_path: Path) -> None:
    story_dict = {
        "acts": _six_acts(),
        "frameworks": _frameworks(entries=[_framework(id="eu-ai-act", type="guideline")]),
        "claims": [_claim(id="claim-one", act=1)],
    }
    path = _write_story(tmp_path, story_dict)

    with pytest.raises(ValueError):
        load_story(path)


# --- coverage_by_framework -------------------------------------------


def test_coverage_reports_backing_claims_per_control_in_declaration_order(tmp_path: Path) -> None:
    from bossyk_sandbox.story import coverage_by_framework

    story_dict = {
        "acts": _six_acts(),
        "frameworks": _frameworks(
            entries=[
                _framework(id="eu-ai-act", controls=[_control("art-12"), _control("art-14")]),
                _framework(
                    id="soc2", name="SOC 2", type="attestation", controls=[_control("cc7-3")]
                ),
            ]
        ),
        "claims": [
            _claim(id="first-claim", act=1, control_refs=["eu-ai-act:art-12", "soc2:cc7-3"]),
            _claim(id="second-claim", act=2, control_refs=["eu-ai-act:art-12"]),
        ],
    }
    story = load_story(_write_story(tmp_path, story_dict))

    coverage = coverage_by_framework(story)

    assert [c.framework.id for c in coverage] == ["eu-ai-act", "soc2"]
    eu_controls = coverage[0].controls
    assert [c.control_id for c in eu_controls] == ["art-12", "art-14"]
    assert [b.claim_id for b in eu_controls[0].backing] == ["first-claim", "second-claim"]
    assert eu_controls[1].backing == []
    assert [b.claim_id for b in coverage[1].controls[0].backing] == ["first-claim"]


def test_coverage_carries_the_evidence_grade_of_each_backing_claim(tmp_path: Path) -> None:
    from bossyk_sandbox.story import EvidenceGrade, coverage_by_framework

    story_dict = {
        "acts": _six_acts(),
        "frameworks": _frameworks(),
        "claims": [
            _claim(
                id="recompute-claim",
                act=1,
                evidence_grade="deterministic-recompute",
                control_refs=["eu-ai-act:art-12"],
            )
        ],
    }
    story = load_story(_write_story(tmp_path, story_dict))

    coverage = coverage_by_framework(story)

    backing = coverage[0].controls[0].backing[0]
    assert backing.claim_id == "recompute-claim"
    assert backing.evidence_grade == EvidenceGrade.DETERMINISTIC_RECOMPUTE


def test_a_control_backed_only_by_an_open_claim_is_not_covered(tmp_path: Path) -> None:
    # The anti-coverage-theatre rule: an open claim is a promise, not evidence.
    from bossyk_sandbox.story import coverage_by_framework

    story_dict = {
        "acts": _six_acts(),
        "frameworks": _frameworks(),
        "claims": [
            _claim(id="open-claim", act=1, evidence_grade="open", control_refs=["eu-ai-act:art-12"])
        ],
    }
    story = load_story(_write_story(tmp_path, story_dict))

    control = coverage_by_framework(story)[0].controls[0]

    assert [b.claim_id for b in control.backing] == ["open-claim"]
    assert control.is_covered is False


def test_a_control_with_one_graded_claim_among_open_ones_is_covered(tmp_path: Path) -> None:
    from bossyk_sandbox.story import coverage_by_framework

    story_dict = {
        "acts": _six_acts(),
        "frameworks": _frameworks(),
        "claims": [
            _claim(
                id="open-claim", act=1, evidence_grade="open", control_refs=["eu-ai-act:art-12"]
            ),
            _claim(
                id="measured-claim",
                act=2,
                evidence_grade="live-measurement",
                control_refs=["eu-ai-act:art-12"],
            ),
        ],
    }
    story = load_story(_write_story(tmp_path, story_dict))

    assert coverage_by_framework(story)[0].controls[0].is_covered is True


def test_coverage_names_the_uncovered_controls(tmp_path: Path) -> None:
    from bossyk_sandbox.story import coverage_by_framework

    story_dict = {
        "acts": _six_acts(),
        "frameworks": _frameworks(
            entries=[
                _framework(
                    id="eu-ai-act",
                    controls=[_control("art-12"), _control("art-14"), _control("art-15")],
                )
            ]
        ),
        "claims": [
            _claim(id="measured-claim", act=1, control_refs=["eu-ai-act:art-12"]),
            _claim(
                id="open-claim", act=2, evidence_grade="open", control_refs=["eu-ai-act:art-14"]
            ),
        ],
    }
    story = load_story(_write_story(tmp_path, story_dict))

    uncovered = coverage_by_framework(story)[0].uncovered

    assert [c.control_id for c in uncovered] == ["art-14", "art-15"]


def test_coverage_of_a_story_without_frameworks_is_empty(tmp_path: Path) -> None:
    from bossyk_sandbox.story import coverage_by_framework

    story_dict = {"acts": _six_acts(), "claims": [_claim(id="untagged-claim", act=1)]}
    story = load_story(_write_story(tmp_path, story_dict))

    assert coverage_by_framework(story) == []


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


def test_cli_prints_the_framework_coverage_report(tmp_path: Path) -> None:
    # The coverage report is the whole point of P2: story_lint must show
    # which regulatory controls the story's claims actually back, and name
    # the uncovered ones rather than hiding a thin mapping behind a tick.
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "scripts/story_lint.py"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    output = result.stdout + result.stderr

    assert "EU AI Act" in output
    assert "SOC 2" in output
    # A control the evidence pack does not discharge must be named as
    # uncovered, not silently omitted -- the anti-coverage-theatre rule.
    assert "uncovered" in output.lower()
    assert "smcr" in output
