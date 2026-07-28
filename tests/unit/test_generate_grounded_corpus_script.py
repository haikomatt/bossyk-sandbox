from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT_PATH = Path(__file__).parent.parent.parent / "scripts" / "generate_grounded_corpus.py"

# Same contract as test_live_h2h4_bench_script: importing the generator
# script must be side-effect-free -- no network, no billable adversary call.
# Only module-level defs/constants run at import; main() is never called
# here (it is gated behind RUN_GROUNDED_GEN=1 + FIREWORKS_API_KEY).


def _import_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("generate_grounded_corpus", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_generate_grounded_corpus_script_imports_without_network_and_defines_main() -> None:
    module = _import_script()

    assert callable(module.main)
    # retail-primary default, overridable via GROUNDED_DOMAIN.
    assert module.DOMAIN in {"airline", "retail"}


def test_output_path_single_mode_is_the_domain_corpus(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GROUNDED_MODE", raising=False)
    module = _import_script()

    assert module._output_path("retail").name == "retail.json"


def test_output_path_goal_mode_is_the_multiturn_corpus(monkeypatch: pytest.MonkeyPatch) -> None:
    # GROUNDED_MODE=goal generates path-B multi-turn user GOALS into a separate
    # corpus, leaving the single-turn (path-A) corpus intact.
    monkeypatch.setenv("GROUNDED_MODE", "goal")
    module = _import_script()

    assert module._output_path("retail").name == "retail-multiturn.json"


def test_output_path_structural_mode_is_the_structural_corpus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # GROUNDED_MODE=structural: single-turn grounded payloads over the STRUCTURAL
    # boundaries only (oracle-tool-pinned, full args), for the dir-1 weakened-agent
    # run -- a separate corpus, leaving the path-A corpus intact.
    monkeypatch.setenv("GROUNDED_MODE", "structural")
    module = _import_script()

    assert module._output_path("retail").name == "retail-structural.json"
