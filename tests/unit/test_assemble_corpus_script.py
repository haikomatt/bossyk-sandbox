"""Hermetic tests for scripts/assemble_corpus.py's corpus-versioning CLI
wiring (hermetic diversity fix, phase-detector-training-step2-datagen.md
Issues & Fixes item 3): --corpus-version, the versioned default path scheme,
and the "refuse to silently mix corpus versions" guard. Pure/offline -- no
network, no LLM.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

from bossyk_sandbox.interp.corpus_assembly import corpus_data_dir

SCRIPT_PATH = Path(__file__).parent.parent.parent / "scripts" / "assemble_corpus.py"


def _import_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("assemble_corpus_cli_test", SCRIPT_PATH)
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


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_arg_parser_defaults_corpus_version_v1() -> None:
    module = _import_script()
    args = module._build_arg_parser().parse_args(["--domain", "retail"])
    assert args.corpus_version == "v1"


def test_default_paths_v1_are_the_flat_legacy_layout(tmp_path: Path) -> None:
    module = _import_script()
    decisions_path, out_dir = module._default_paths(tmp_path, "retail", "v1")
    assert decisions_path == corpus_data_dir(tmp_path, "retail", "v1") / "decisions.jsonl"
    assert decisions_path == tmp_path / "retail" / "decisions.jsonl"
    assert out_dir == tmp_path / "retail"


def test_default_paths_other_versions_get_a_subdirectory(tmp_path: Path) -> None:
    module = _import_script()
    decisions_path, out_dir = module._default_paths(tmp_path, "retail", "v2")
    assert decisions_path == tmp_path / "retail" / "v2" / "decisions.jsonl"
    assert out_dir == tmp_path / "retail" / "v2"


def test_main_refuses_a_decisions_file_with_internally_mixed_versions(tmp_path: Path) -> None:
    module = _import_script()
    decisions_path = tmp_path / "decisions.jsonl"
    _write_jsonl(
        decisions_path,
        [_row("s1", "a", corpus_version="v1"), _row("s2", "b", corpus_version="v2")],
    )
    rc = module.main(
        ["--domain", "retail", "--decisions-path", str(decisions_path), "--out-dir", str(tmp_path)]
    )
    assert rc == 1
    assert not (tmp_path / "train.jsonl").exists()


def test_main_refuses_when_file_version_does_not_match_requested(tmp_path: Path) -> None:
    module = _import_script()
    decisions_path = tmp_path / "decisions.jsonl"
    _write_jsonl(decisions_path, [_row("s1", "a", corpus_version="v2")])
    rc = module.main(
        [
            "--domain",
            "retail",
            "--decisions-path",
            str(decisions_path),
            "--out-dir",
            str(tmp_path),
            "--corpus-version",
            "v1",  # mismatch: the file says v2
        ]
    )
    assert rc == 1
    assert not (tmp_path / "train.jsonl").exists()


def test_main_succeeds_when_versions_agree(tmp_path: Path) -> None:
    module = _import_script()
    decisions_path = tmp_path / "decisions.jsonl"
    rows = [_row(f"s{i}", f"do thing {i}", corpus_version="v2") for i in range(6)]
    _write_jsonl(decisions_path, rows)
    rc = module.main(
        [
            "--domain",
            "retail",
            "--decisions-path",
            str(decisions_path),
            "--out-dir",
            str(tmp_path),
            "--corpus-version",
            "v2",
        ]
    )
    assert rc == 0
    assert (tmp_path / "train.jsonl").exists()


def test_main_succeeds_on_run_1_style_unversioned_rows_with_default_v1(tmp_path: Path) -> None:
    # Run-1's already-committed decisions.jsonl carries NO corpus_version key
    # at all -- the default --corpus-version="v1" must still accept it.
    module = _import_script()
    decisions_path = tmp_path / "decisions.jsonl"
    rows = [_row(f"s{i}", f"do thing {i}") for i in range(6)]  # no corpus_version
    _write_jsonl(decisions_path, rows)
    rc = module.main(
        ["--domain", "retail", "--decisions-path", str(decisions_path), "--out-dir", str(tmp_path)]
    )
    assert rc == 0
    assert (tmp_path / "train.jsonl").exists()
