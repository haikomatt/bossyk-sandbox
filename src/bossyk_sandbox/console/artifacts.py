"""Read-only artifact API for the bossyk-sandbox evidence browser (F1
backend). An `APIRouter` mounted into the existing console `app`
(`console/app.py`) -- it never touches session/WS state, only the repo's
committed artifacts on disk: `story/story.yaml`, `docs/bench_output/*.json`,
`probes/regression/*.json`, `docs/figures/*.{svg,png}`, `docs/*.md`, and
(optionally absent) `demo_output/*.pack.json`.

Every path is resolved through `_repo_root()` at call time -- never a
module-level constant -- both because that mirrors `env.bossyk_root()`'s
call-time-resolution convention elsewhere in this repo, and because it gives
tests a single seam (`monkeypatch.setattr(artifacts, "_repo_root", ...)`) to
point the whole router at a tmp fixture tree.

Path safety: a requested `name` is only ever joined onto a directory path
after (a) rejecting outright anything containing `/`, `\\`, or `..`, and (b)
confirming it appears verbatim in that directory's actual `os.listdir()`
result. This order matters -- the listing check must not itself join
untrusted input into a path, so it lists the (trusted, server-side) directory
and checks membership, rather than constructing a candidate path first and
testing whether it exists."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import yaml
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from bossyk_sandbox.story import load_story

router = APIRouter()

_MANIFEST_HEADER = "## Reproducibility manifest"
_NEXT_HEADING_RE = re.compile(r"\n##[^#]")
_YAML_FENCE_RE = re.compile(r"```yaml\n(.*?)```", re.DOTALL)

# Repo-relative (dir_parts, glob_suffixes) for each JSON-passthrough
# artifact category. `figures` and `docs` are handled separately below since
# they don't just parse-and-return JSON.
_JSON_CATEGORIES: dict[str, tuple[str, ...]] = {
    "bench_output": ("docs", "bench_output"),
    "probes": ("probes", "regression"),
    "packs": ("demo_output",),
}


def _repo_root() -> Path:
    """`src/bossyk_sandbox/console/artifacts.py` -> repo root is four
    parents up. A function (not a constant) so tests can monkeypatch it to
    point the whole router at a tmp fixture tree."""
    return Path(__file__).resolve().parent.parent.parent.parent


def _sorted_names(dir_path: Path, suffixes: tuple[str, ...]) -> list[str]:
    """Sorted filenames directly inside `dir_path` matching any of
    `suffixes`, skipping subdirectories. An absent `dir_path` (e.g.
    `demo_output/` in a checkout with no demo packs yet) is `[]`, never an
    error."""
    if not dir_path.is_dir():
        return []
    return sorted(
        name
        for name in os.listdir(dir_path)
        if name.endswith(suffixes) and (dir_path / name).is_file()
    )


def _safe_path_or_404(dir_path: Path, name: str) -> Path:
    """Resolves `name` inside `dir_path`, or raises 404. Rejects any name
    containing a path separator or `..` before doing anything else; only
    then checks membership in `os.listdir(dir_path)` (never existence of a
    joined path) and that the resulting entry is a regular file."""
    if not name or "/" in name or "\\" in name or ".." in name:
        raise HTTPException(status_code=404, detail="not found")
    if not dir_path.is_dir() or name not in os.listdir(dir_path):
        raise HTTPException(status_code=404, detail="not found")
    candidate = dir_path / name
    if not candidate.is_file():
        raise HTTPException(status_code=404, detail="not found")
    return candidate


def _manifest_section(text: str) -> str | None:
    """The text from `## Reproducibility manifest` up to (but excluding)
    the next `##` heading, or `None` if the doc has no such section."""
    idx = text.find(_MANIFEST_HEADER)
    if idx == -1:
        return None
    rest = text[idx + len(_MANIFEST_HEADER) :]
    next_heading = _NEXT_HEADING_RE.search(rest)
    return rest[: next_heading.start()] if next_heading else rest


@router.get("/api/story")
async def get_story() -> dict[str, Any]:
    story = load_story(_repo_root() / "story" / "story.yaml")
    return story.model_dump(mode="json")


@router.get("/api/artifacts")
async def list_artifacts() -> dict[str, list[str]]:
    root = _repo_root()
    return {
        "bench_output": _sorted_names(root / "docs" / "bench_output", (".json",)),
        "probes": _sorted_names(root / "probes" / "regression", (".json",)),
        "figures": _sorted_names(root / "docs" / "figures", (".svg", ".png")),
        "docs": _sorted_names(root / "docs", (".md",)),
        "packs": _sorted_names(root / "demo_output", (".pack.json",)),
    }


@router.get("/api/artifacts/{category}/{name}")
async def get_artifact(category: str, name: str) -> Any:
    root = _repo_root()

    if category in _JSON_CATEGORIES:
        dir_path = root
        for part in _JSON_CATEGORIES[category]:
            dir_path = dir_path / part
        path = _safe_path_or_404(dir_path, name)
        return json.loads(path.read_text())

    if category == "figures":
        path = _safe_path_or_404(root / "docs" / "figures", name)
        media_type = "image/svg+xml" if path.suffix == ".svg" else "image/png"
        return FileResponse(path, media_type=media_type)

    if category == "docs":
        path = _safe_path_or_404(root / "docs", name)
        return {"name": name, "markdown": path.read_text()}

    raise HTTPException(status_code=404, detail=f"unknown artifact category {category!r}")


@router.get("/api/manifests")
async def get_manifests() -> list[dict[str, Any]]:
    root = _repo_root()
    docs_dir = root / "docs"
    results: list[dict[str, Any]] = []
    for name in _sorted_names(docs_dir, (".md",)):
        section = _manifest_section((docs_dir / name).read_text())
        if section is None:
            continue
        manifests = [yaml.safe_load(block) for block in _YAML_FENCE_RE.findall(section)]
        if manifests:
            results.append({"doc": name, "manifests": manifests})
    return results
