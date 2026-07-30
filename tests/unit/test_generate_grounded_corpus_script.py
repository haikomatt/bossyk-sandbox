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


# --- bossyk-sandbox slice 3, phase 3a (RED): outreach live-path wiring ------
def test_tool_schemas_by_domain_will_include_outreach() -> None:
    module = _import_script()

    assert "outreach" in module._TOOL_SCHEMAS_BY_DOMAIN


# --- bossyk-sandbox slice 3 bugfix (RED): grounded grid restriction --------
# The live 3b run crashed: GROUNDED_MODE=single (the default) built cells
# from `boundaries_for(domain)` (the FULL named-boundary taxonomy), but
# fireworks_adversary.py's prompt-building calls `boundary_spec_for` per
# cell whenever tool_context is supplied (always true for GROUNDED runs) --
# raising KeyError for any boundary with no registered BoundarySpec.
# Outreach's out_of_hours_contact (cell 2) crashed generation ~10 calls in.


def test_single_mode_cells_use_grounded_boundaries_not_the_full_named_grid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GROUNDED_MODE", raising=False)
    module = _import_script()

    cells = module._cells("outreach")
    boundaries = {cell.boundary for cell in cells}

    assert boundaries == {
        "contact_without_consent",
        "booking_without_eligibility",
        "unauthorised_incentive",
    }
    assert "out_of_hours_contact" not in boundaries
    assert "prohibited_financial_promotion" not in boundaries
    assert "pii_cross_contact" not in boundaries


def test_single_mode_cells_all_resolve_a_boundary_spec_for_every_registered_domain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # THE regression guard that would have caught the crash: every cell the
    # single-mode grid produces, for every domain the grounded generator
    # actually supports, must resolve boundary_spec_for without raising --
    # exactly the call fireworks_adversary.py makes per cell.
    from bossyk_sandbox.conditions.live_boundary import boundary_spec_for

    monkeypatch.delenv("GROUNDED_MODE", raising=False)
    module = _import_script()

    for domain in module._TOOL_SCHEMAS_BY_DOMAIN:
        cells = module._cells(domain)
        assert cells, f"{domain} grid is empty"
        for cell in cells:
            boundary_spec_for(cell.boundary, domain=cell.domain)  # must not raise


def test_goal_and_structural_mode_cells_also_all_resolve_a_boundary_spec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # goal/structural already use structural_boundaries (a subset of
    # spec-backed boundaries), so this should already hold -- a defensive
    # regression guard against a future drift in that path too.
    from bossyk_sandbox.conditions.live_boundary import boundary_spec_for

    for mode in ("goal", "structural"):
        monkeypatch.setenv("GROUNDED_MODE", mode)
        module = _import_script()
        for domain in module._TOOL_SCHEMAS_BY_DOMAIN:
            for cell in module._cells(domain):
                boundary_spec_for(cell.boundary, domain=cell.domain)  # must not raise


def test_single_mode_cells_for_retail_are_unchanged_by_the_restriction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # No-op proof at the script level (mirrors test_grounded_boundaries.py's
    # unit-level proof): retail's grid is identical before/after the fix.
    from bossyk_sandbox.conditions.grid import AttackClass, boundaries_for, build_grid

    monkeypatch.delenv("GROUNDED_MODE", raising=False)
    module = _import_script()

    assert module._cells("retail") == build_grid(
        "retail", list(AttackClass), boundaries_for("retail")
    )


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
