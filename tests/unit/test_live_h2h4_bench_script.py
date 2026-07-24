from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT_PATH = Path(__file__).parent.parent.parent / "scripts" / "live_h2h4_bench.py"

# scripts/ isn't a package (no __init__.py, not in pyproject's packages),
# so this loads the module by file path -- mirroring how `uv run python
# scripts/live_h2h4_bench.py` would import it, without going through
# `import scripts.live_h2h4_bench`. Importing it must be side-effect-free
# (no network, no billable call): only module-level defs/constants run at
# import time, main() is never called here.


def _import_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("live_h2h4_bench", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_live_h2h4_bench_script_imports_without_network_and_defines_main() -> None:
    module = _import_script()

    assert callable(module.main)
    # retail-primary: the default domain is retail unless LIVE_H2_DOMAIN
    # overrides it.
    assert module.DOMAIN in {"airline", "retail"}
    assert set(module._RUN_SESSION_BY_DOMAIN) == {"airline", "retail"}


def test_corpus_path_defaults_to_the_per_domain_regression_file() -> None:
    module = _import_script()

    assert module._corpus_path("retail") == module.REGRESSION_PROBES_DIR / "retail.json"


def test_corpus_path_honors_the_live_h2_corpus_override(monkeypatch: pytest.MonkeyPatch) -> None:
    # The grounded run reuses this bench but points it at the grounded
    # corpus (probes/grounded/retail.json) instead of the H1 artifact --
    # LIVE_H2_CORPUS is the seam that swaps the input without a code change.
    monkeypatch.setenv("LIVE_H2_CORPUS", "/tmp/grounded/retail.json")
    module = _import_script()

    assert module._corpus_path("retail") == Path("/tmp/grounded/retail.json")


def test_output_path_defaults_to_the_per_domain_bench_output() -> None:
    module = _import_script()

    assert module._output_path("retail") == module.OUTPUT_DIR / "live_h2h4_retail.json"


def test_output_path_honors_the_live_h2_output_override(monkeypatch: pytest.MonkeyPatch) -> None:
    # The grounded run must NOT clobber the committed L1 artifact
    # (live_h2h4_retail.json); LIVE_H2_OUTPUT redirects it to a distinct file
    # so both the L1 and grounded results survive for the before/after.
    monkeypatch.setenv("LIVE_H2_OUTPUT", "/tmp/grounded/out.json")
    module = _import_script()

    assert module._output_path("retail") == Path("/tmp/grounded/out.json")


def test_live_h2_mode_defaults_to_single_turn() -> None:
    module = _import_script()

    assert module.LIVE_H2_MODE == "single"
    assert module.LIVE_H2_MAX_TURNS >= 1


def test_run_session_for_single_mode_is_the_domain_runner() -> None:
    module = _import_script()

    assert module._run_session_for("retail") is module._RUN_SESSION_BY_DOMAIN["retail"]


def test_run_session_for_multiturn_retail_is_a_callable(monkeypatch: pytest.MonkeyPatch) -> None:
    # LIVE_H2_MODE=multiturn drives path B: each corpus probe's payload is the
    # adversarial GOAL, replayed as a full multi-turn episode.
    monkeypatch.setenv("LIVE_H2_MODE", "multiturn")
    module = _import_script()

    assert callable(module._run_session_for("retail"))


def test_live_h2_agent_defaults_to_compliant() -> None:
    module = _import_script()

    assert module.LIVE_H2_AGENT == "compliant"


def test_run_session_for_weak_agent_is_the_weakened_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # dir 1: LIVE_H2_AGENT=weak replays the corpus against an under-specified
    # (weakened-policy) retail agent, so the gate's prevention leg fires live.
    monkeypatch.setenv("LIVE_H2_AGENT", "weak")
    module = _import_script()

    assert module._run_session_for("retail") is module.run_live_weakened_retail_session


# --- latency budget wiring (§15B+) -------------------------------------------


def test_mean_action_exec_s_averages_executed_tool_latency() -> None:
    module = _import_script()

    records = [
        module.LatencyRecord(instrument="action_exec", elapsed_s=0.002),
        module.LatencyRecord(instrument="action_exec", elapsed_s=0.004),
    ]

    assert module._mean_action_exec_s(records) == pytest.approx(0.003)


def test_mean_action_exec_s_is_none_when_no_tool_executed() -> None:
    # Every proposed action gate-blocked pre-execution -> nothing to time ->
    # the budget reports the UX sweep alone, not a fabricated floor.
    module = _import_script()

    assert module._mean_action_exec_s([]) is None


def test_latency_budget_to_dict_serializes_the_per_detector_speedup_target() -> None:
    module = _import_script()

    summaries = {
        "policy": module.LatencySummary(
            instrument="policy", count=4, mean_s=26.0, p50_s=42.0, p95_s=42.0, max_s=42.0
        )
    }
    budget = module.latency_budget(summaries, action_exec_s=None, ux_budgets_s={"ux_500ms": 0.5})

    serialized = module._latency_budget_to_dict(budget)

    assert set(serialized) == {"policy"}
    row = serialized["policy"][0]
    assert row["budget_label"] == "ux_500ms"
    # 26s / 0.5s = 52x (mean), 42s / 0.5s = 84x (p95) -- the §15C target.
    assert row["speedup_needed_mean"] == pytest.approx(52.0)
    assert row["speedup_needed_p95"] == pytest.approx(84.0)
