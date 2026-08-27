"""Hermetic tests for scripts/generate_corpus.py's CLI wiring (hermetic
diversity fix, phase-detector-training-step2-datagen.md Issues & Fixes):
--temperature/--top-p sampling-param plumbing and --corpus-version
versioned-path wiring. Every run here uses --stub (a scripted LLM that
replays each scenario's own authored tool-call sequence) -- zero network,
zero API key, nothing billable. Stub-tests the PLUMBING, not the sampling
itself: temperature/top_p are unused once an `llm` is already injected (see
langgraph_agent._build_agent_session), so this proves the CLI -> driver ->
builder wiring without ever constructing a real ChatOpenAI client.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

from bossyk_sandbox.interp.corpus_assembly import corpus_data_dir
from bossyk_sandbox.interp.datagen_driver import load_checkpoint

SCRIPT_PATH = Path(__file__).parent.parent.parent / "scripts" / "generate_corpus.py"


def _import_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("generate_corpus_cli_test", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # StubScenarioLLM is a @dataclass under `from __future__ import
    # annotations` -- its string-annotation resolution needs the module
    # registered in sys.modules BEFORE exec_module (same gotcha as
    # test_datagen_pipeline_smoke.py).
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_imports_side_effect_free_and_defines_main() -> None:
    module = _import_script()
    assert callable(module.main)


# --- CLI defaults: byte-identical when the new flags are unset --------------


def test_arg_parser_defaults_temperature_0_top_p_none_corpus_version_v1() -> None:
    module = _import_script()
    args = module._build_arg_parser().parse_args(["--domain", "retail", "--stub"])
    assert args.temperature == 0.0
    assert args.top_p is None
    assert args.corpus_version == "v1"


def test_arg_parser_accepts_the_new_flags() -> None:
    module = _import_script()
    args = module._build_arg_parser().parse_args(
        [
            "--domain",
            "retail",
            "--stub",
            "--temperature",
            "0.8",
            "--top-p",
            "0.9",
            "--corpus-version",
            "v2",
        ]
    )
    assert args.temperature == 0.8
    assert args.top_p == 0.9
    assert args.corpus_version == "v2"


# --- _session_kwargs_from_args: the plumbing, stub-tested --------------------


def test_session_kwargs_from_args_omits_top_p_when_unset() -> None:
    module = _import_script()
    args = module._build_arg_parser().parse_args(["--domain", "retail", "--stub"])
    assert module._session_kwargs_from_args(args) == {"temperature": 0.0}


def test_session_kwargs_from_args_includes_top_p_when_set() -> None:
    module = _import_script()
    args = module._build_arg_parser().parse_args(
        ["--domain", "retail", "--stub", "--temperature", "0.7", "--top-p", "0.4"]
    )
    assert module._session_kwargs_from_args(args) == {"temperature": 0.7, "top_p": 0.4}


# --- default checkpoint path: versioned subdirectory scheme ------------------


def test_default_checkpoint_path_v1_is_the_flat_legacy_path() -> None:
    module = _import_script()
    path = module._default_checkpoint_path("retail", "v1")
    assert path == corpus_data_dir(module.DATA_ROOT, "retail", "v1") / "decisions.jsonl"
    assert path == module.DATA_ROOT / "retail" / "decisions.jsonl"  # run-1's actual layout


def test_default_checkpoint_path_other_versions_get_a_subdirectory() -> None:
    module = _import_script()
    path = module._default_checkpoint_path("retail", "v2")
    assert path == module.DATA_ROOT / "retail" / "v2" / "decisions.jsonl"


# --- end-to-end stub run: zero network, proves the full CLI wiring ----------


def test_stub_run_default_flags_stamps_corpus_version_v1(tmp_path: Path) -> None:
    module = _import_script()
    ckpt = tmp_path / "decisions.jsonl"
    rc = module.main(
        ["--domain", "retail", "--stub", "--variants-per-scenario", "1", "--checkpoint", str(ckpt)]
    )
    assert rc == 0
    _, rows = load_checkpoint(ckpt)
    assert rows
    assert all(r["corpus_version"] == "v1" for r in rows)


def test_stub_run_with_new_flags_stamps_the_requested_corpus_version(tmp_path: Path) -> None:
    module = _import_script()
    ckpt = tmp_path / "decisions.jsonl"
    rc = module.main(
        [
            "--domain",
            "retail",
            "--stub",
            "--variants-per-scenario",
            "1",
            "--checkpoint",
            str(ckpt),
            "--temperature",
            "0.6",
            "--top-p",
            "0.5",
            "--corpus-version",
            "v2",
        ]
    )
    assert rc == 0
    _, rows = load_checkpoint(ckpt)
    assert rows
    assert all(r["corpus_version"] == "v2" for r in rows)
