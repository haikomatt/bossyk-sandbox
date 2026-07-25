from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from bossyk_sandbox.console import app as console_app
from bossyk_sandbox.console import artifacts as artifacts_module

client = TestClient(console_app.app)


# --- story ------------------------------------------------------------------


def test_story_endpoint_returns_full_story() -> None:
    resp = client.get("/api/story")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["acts"]) == 6
    # 20 since P3 added attested-action-log-is-the-substrate (the sample
    # evidence pack claim that anchors the logging/attestation controls).
    assert len(body["claims"]) == 20
    claim = body["claims"][0]
    for field in (
        "id",
        "act",
        "exec_copy",
        "tech_copy",
        "verdict",
        "evidence_grade",
        "artifact_refs",
        "figure_ids",
        "numeric_checks",
        "control_refs",
    ):
        assert field in claim


def test_story_endpoint_serves_the_frameworks_catalogue() -> None:
    # The compliance axis must survive serialization so the SPA and the deck
    # can render framework tags and coverage from one source of truth.
    resp = client.get("/api/story")
    body = resp.json()

    frameworks = body["frameworks"]
    assert frameworks is not None
    assert "disclaimer" in frameworks
    ids = [entry["id"] for entry in frameworks["entries"]]
    assert {"eu-ai-act", "hipaa", "soc2"} <= set(ids)
    eu = next(entry for entry in frameworks["entries"] if entry["id"] == "eu-ai-act")
    assert {"id", "ref", "title"} <= set(eu["controls"][0])

    # At least one claim carries a resolvable control_ref into that catalogue.
    tagged = [claim for claim in body["claims"] if claim["control_refs"]]
    assert tagged
    assert ":" in tagged[0]["control_refs"][0]


def test_frameworks_endpoint_serves_the_catalogue() -> None:
    # The evidence-browser trace view resolves control refs to human names
    # through this endpoint, independently of the story.
    resp = client.get("/api/frameworks")
    assert resp.status_code == 200
    body = resp.json()

    assert "not legal advice" in body["disclaimer"].lower()
    ids = [entry["id"] for entry in body["entries"]]
    assert {"eu-ai-act", "hipaa", "soc2"} <= set(ids)
    assert {"id", "ref", "title"} <= set(body["entries"][0]["controls"][0])


# --- listing ------------------------------------------------------------------


def test_artifacts_listing_includes_known_real_files() -> None:
    resp = client.get("/api/artifacts")
    assert resp.status_code == 200
    body = resp.json()
    assert "phase3_h4.json" in body["bench_output"]
    assert "retail-smactr.json" in body["probes"]
    assert body["bench_output"] == sorted(body["bench_output"])
    assert body["probes"] == sorted(body["probes"])
    assert body["figures"] == sorted(body["figures"])
    assert body["docs"] == sorted(body["docs"])
    assert body["packs"] == []


# --- json artifact round-trip -------------------------------------------------


def test_bench_output_json_roundtrips(tmp_path: Path) -> None:
    resp = client.get("/api/artifacts/bench_output/phase3_h4.json")
    assert resp.status_code == 200
    from_disk = (
        artifacts_module._repo_root() / "docs" / "bench_output" / "phase3_h4.json"
    ).read_text()
    import json

    assert resp.json() == json.loads(from_disk)


def test_probes_json_roundtrips() -> None:
    resp = client.get("/api/artifacts/probes/retail-smactr.json")
    assert resp.status_code == 200
    from_disk = (
        artifacts_module._repo_root() / "probes" / "regression" / "retail-smactr.json"
    ).read_text()
    import json

    assert resp.json() == json.loads(from_disk)


# --- figures -------------------------------------------------------------------


def test_figure_fetch_svg_content_type() -> None:
    resp = client.get("/api/artifacts/figures/h1-blindspot-by-class.svg")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("image/svg+xml")


def test_figure_fetch_png_content_type() -> None:
    resp = client.get("/api/artifacts/figures/h1-blindspot-by-class.png")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("image/png")


# --- docs ----------------------------------------------------------------------


def test_docs_fetch_returns_markdown_text() -> None:
    resp = client.get("/api/artifacts/docs/phase3-h4-interrupt-results.md")
    assert resp.status_code == 200
    body = resp.json()
    assert body["name"] == "phase3-h4-interrupt-results.md"
    assert "#" in body["markdown"]


# --- packs -----------------------------------------------------------------------


def test_packs_fetch_404_when_absent() -> None:
    resp = client.get("/api/artifacts/packs/anything.pack.json")
    assert resp.status_code == 404


# --- manifests -----------------------------------------------------------------


def test_manifests_endpoint_finds_docs_with_script_key() -> None:
    resp = client.get("/api/manifests")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) >= 6
    for entry in body:
        assert "doc" in entry
        assert len(entry["manifests"]) >= 1
        for manifest in entry["manifests"]:
            assert "script" in manifest


# --- 404s ------------------------------------------------------------------------


def test_unknown_category_is_404() -> None:
    resp = client.get("/api/artifacts/not-a-real-category/foo.json")
    assert resp.status_code == 404


def test_unknown_name_is_404() -> None:
    resp = client.get("/api/artifacts/bench_output/does-not-exist.json")
    assert resp.status_code == 404


# --- traversal -------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "/api/artifacts/bench_output/../../pyproject.toml",
        "/api/artifacts/bench_output/%2e%2e%2f%2e%2e%2fpyproject.toml",
        "/api/artifacts/docs/..%2f..%2fpyproject.toml",
    ],
)
def test_path_traversal_is_rejected(path: str) -> None:
    resp = client.get(path)
    assert resp.status_code in (404, 400)


def test_path_safety_checks_against_actual_directory_listing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The requested name must appear in os.listdir of the real directory --
    not merely resolve to something on disk. Point _repo_root at a tmp tree
    with a decoy file elsewhere on disk and confirm it's unreachable."""
    fake_root = tmp_path / "fake_repo"
    (fake_root / "docs" / "bench_output").mkdir(parents=True)
    (fake_root / "docs" / "bench_output" / "real.json").write_text('{"ok": true}')
    secret = tmp_path / "secret.json"
    secret.write_text('{"leaked": true}')

    monkeypatch.setattr(artifacts_module, "_repo_root", lambda: fake_root)

    ok = client.get("/api/artifacts/bench_output/real.json")
    assert ok.status_code == 200
    assert ok.json() == {"ok": True}

    leaked = client.get("/api/artifacts/bench_output/../secret.json")
    assert leaked.status_code == 404


def test_console_session_tests_still_pass_smoke() -> None:
    """Sanity: mounting the artifacts router did not disturb the existing
    session endpoints."""
    resp = client.post("/session/decide", json={"session_id": "nope", "verdict": "block"})
    assert resp.status_code == 200
    assert resp.json() == {"status": "unknown_session"}
