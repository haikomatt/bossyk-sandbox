"""Hermetic tests for scripts/qc_corpus.py's corpus-versioning + sampling-
param CLI wiring (hermetic diversity fix, phase-detector-training-step2-
datagen.md Issues & Fixes items 1 and 3): --corpus-version/--temperature/
--top-p, the versioned default data-dir scheme, the "refuse to silently mix
corpus versions" guard, and the manifest picking those values up. Pure/
offline -- no network, no LLM.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

SCRIPT_PATH = Path(__file__).parent.parent.parent / "scripts" / "qc_corpus.py"


def _import_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("qc_corpus_cli_test", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(scenario_id: str, prompt: str, *, corpus_version: str | None = None) -> dict[str, object]:
    row: dict[str, object] = {
        "row_id": f"retail:{scenario_id}:0#turn-0",
        "task_id": f"retail:{scenario_id}:0",
        "domain": "retail",
        "scenario_id": scenario_id,
        "persona_id": None,
        "variant_index": 0,
        "step_id": "turn-0",
        "prompt": prompt,
        "is_violation": False,
        "action": None,
        "generator": "stub",
        "empty": False,
    }
    if corpus_version is not None:
        row["corpus_version"] = corpus_version
    return row


def _write_split(domain_dir: Path, split: str, rows: list[dict[str, object]]) -> None:
    domain_dir.mkdir(parents=True, exist_ok=True)
    (domain_dir / f"{split}.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + ("\n" if rows else "")
    )


def test_arg_parser_defaults() -> None:
    module = _import_script()
    args = module._build_arg_parser().parse_args(["--domains", "retail"])
    assert args.corpus_version == "v1"
    assert args.temperature == 0.0
    assert args.top_p is None


def test_main_refuses_a_domain_with_internally_mixed_versions(tmp_path: Path) -> None:
    module = _import_script()
    domain_dir = tmp_path / "retail"
    _write_split(domain_dir, "train", [_row("s1", "a", corpus_version="v1")])
    _write_split(domain_dir, "val", [_row("s2", "b", corpus_version="v2")])
    _write_split(domain_dir, "test", [])
    rc = module.main(["--domains", "retail", "--data-root", str(tmp_path)])
    assert rc == 1
    assert not (domain_dir / "manifest.json").exists()


def test_main_refuses_when_domain_version_does_not_match_requested(tmp_path: Path) -> None:
    module = _import_script()
    domain_dir = tmp_path / "retail"
    _write_split(domain_dir, "train", [_row("s1", "a", corpus_version="v2")])
    _write_split(domain_dir, "val", [])
    _write_split(domain_dir, "test", [])
    rc = module.main(
        ["--domains", "retail", "--data-root", str(tmp_path), "--corpus-version", "v1"]
    )
    assert rc == 1
    assert not (domain_dir / "manifest.json").exists()


def test_main_writes_manifest_with_corpus_version_and_sampling_params(tmp_path: Path) -> None:
    module = _import_script()
    domain_dir = tmp_path / "retail" / "v2"  # v2 lives in its own subdirectory
    _write_split(domain_dir, "train", [_row("s1", "a", corpus_version="v2")])
    _write_split(domain_dir, "val", [_row("s2", "b", corpus_version="v2")])
    _write_split(domain_dir, "test", [])
    rc = module.main(
        [
            "--domains",
            "retail",
            "--data-root",
            str(tmp_path),
            "--corpus-version",
            "v2",
            "--temperature",
            "0.8",
            "--top-p",
            "0.95",
        ]
    )
    assert rc == 0
    manifest = json.loads((domain_dir / "manifest.json").read_text())
    assert manifest["corpus_version"] == "v2"
    assert manifest["temperature"] == 0.8
    assert manifest["top_p"] == 0.95


def test_main_succeeds_on_run_1_style_unversioned_rows_with_default_v1(tmp_path: Path) -> None:
    module = _import_script()
    domain_dir = tmp_path / "retail"
    _write_split(domain_dir, "train", [_row("s1", "a")])  # no corpus_version key
    _write_split(domain_dir, "val", [_row("s2", "b")])
    _write_split(domain_dir, "test", [])
    rc = module.main(["--domains", "retail", "--data-root", str(tmp_path)])
    assert rc == 0
    manifest = json.loads((domain_dir / "manifest.json").read_text())
    assert manifest["corpus_version"] == "v1"
