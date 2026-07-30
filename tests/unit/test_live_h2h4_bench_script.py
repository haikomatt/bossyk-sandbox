from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

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
    # Test-Integrity note: bossyk-sandbox slice 3, phase 3a registered
    # "outreach" -> run_live_outreach_session (the zero-cost prerequisite
    # for the outreach live run) -- this assumption was deliberately
    # invalidated, not silently broken; see conditions.live_replay's
    # run_live_outreach_session and this file's now-folded-in
    # test_run_session_by_domain_will_include_outreach (RED).
    assert set(module._RUN_SESSION_BY_DOMAIN) == {"airline", "retail", "outreach"}


def test_corpus_path_defaults_to_the_per_domain_regression_file() -> None:
    module = _import_script()

    assert module._corpus_path("retail") == module.REGRESSION_PROBES_DIR / "retail.json"


# --- bossyk-sandbox slice 3 (RED): utterance-line scoreboard dispatch ------
# Matt's decision: pull the 3c utterance-line piece forward so the outreach
# re-run demonstrates boundary 5 too. Proposed dispatch: a new
# `_score_replay(replay, policy_verdicts)` picks `score_crossing` for a
# tool-call boundary (has_boundary_spec resolves) and
# `score_utterance_crossing` otherwise (see test_utterance_crossing_score.py
# for that function's own spec) -- `main()`'s loop would call THIS instead
# of `score_crossing` directly. Not implemented in this RED pass.


def _fake_probe(probe_id: str, family: str) -> Any:
    from auditk.adapters.protocols import Stimulus
    from auditk.schema import ExpectedBehavior, ProbeDefinition

    ProbeDefinition.model_rebuild()
    return ProbeDefinition(
        probe_id=probe_id,
        family=family,
        version="0.1",
        kind="jailbreak",
        stimulus=Stimulus(channel="user", payload={"text": "test"}),
        expected_behavior=ExpectedBehavior(should_refuse=True),
    )


def test_score_replay_dispatches_to_score_crossing_for_a_tool_call_boundary() -> None:
    from bossyk_sandbox.conditions.live_replay import CrossingReplay
    from bossyk_sandbox.instruments.base import ProposedAction
    from bossyk_sandbox.scoring.live_h2 import CrossingScore

    module = _import_script()
    replay = CrossingReplay(
        probe=_fake_probe("retail-cancel_without_auth-jailbreak-0", "retail-cancel_without_auth"),
        domain="retail",
        boundary="cancel_without_auth",
        proposed=[ProposedAction("cancel_pending_order", {"order_id": "#W1"})],
        executed=[],
    )

    score = module._score_replay(replay, [])

    assert isinstance(score, CrossingScore)
    assert score.reached is True


def test_score_replay_dispatches_to_score_utterance_crossing_for_the_utterance_boundary() -> None:
    from bossyk_sandbox.conditions.live_replay import CrossingReplay
    from bossyk_sandbox.instruments.base import Decision, Verdict
    from bossyk_sandbox.scoring.live_h2 import UtteranceCrossingScore

    module = _import_script()
    blocked = Decision(Verdict.BLOCK, "utterance contains regulated phrase '0% finance'")
    replay = CrossingReplay(
        probe=_fake_probe(
            "outreach-prohibited_financial_promotion-jailbreak-0",
            "outreach-prohibited_financial_promotion",
        ),
        domain="outreach",
        boundary="prohibited_financial_promotion",
        proposed=[],
        executed=[],
        utterance_decisions=[blocked],
    )

    score = module._score_replay(replay, [])

    assert isinstance(score, UtteranceCrossingScore)
    assert score.reached is True


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


# --- weak-outreach live-bench wiring (RED): _run_session_for's weak branch
# was hardcoded to run_live_weakened_retail_session regardless of domain --
# build_weakened_outreach_agent_session exists (3a) but nothing wired it
# into the bench. Proposed fix: a _WEAK_SESSION_BY_DOMAIN registry, and the
# weak branch dispatches on it instead of hardcoding retail.


def test_weak_session_by_domain_will_include_outreach() -> None:
    module = _import_script()

    assert "outreach" in module._WEAK_SESSION_BY_DOMAIN


def test_weak_session_by_domain_retail_entry_is_unchanged() -> None:
    module = _import_script()

    assert module._WEAK_SESSION_BY_DOMAIN["retail"] is module.run_live_weakened_retail_session


def test_run_session_for_weak_outreach_resolves_the_weakened_outreach_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LIVE_H2_AGENT", "weak")
    module = _import_script()

    assert module._run_session_for("outreach") is module.run_live_weakened_outreach_session


def test_real_mode_requested_permits_weak_plus_outreach(monkeypatch: pytest.MonkeyPatch) -> None:
    # Hermetic: _real_mode_requested() is a pure env-var guard -- it never
    # makes a network call or touches a real client, so monkeypatching
    # RUN_LIVE_H2_E2E here (auto-reverted after the test) never spends
    # anything, exactly like this file's other env-gated tests.
    monkeypatch.setenv("RUN_LIVE_H2_E2E", "1")
    monkeypatch.setenv("FIREWORKS_API_KEY", "fake-key-hermetic-test-only")
    monkeypatch.setenv("LIVE_H2_DOMAIN", "outreach")
    monkeypatch.setenv("LIVE_H2_AGENT", "weak")
    module = _import_script()

    assert module._real_mode_requested() is True


def test_real_mode_requested_still_rejects_weak_plus_airline_with_a_clear_message(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("RUN_LIVE_H2_E2E", "1")
    monkeypatch.setenv("FIREWORKS_API_KEY", "fake-key-hermetic-test-only")
    monkeypatch.setenv("LIVE_H2_DOMAIN", "airline")
    monkeypatch.setenv("LIVE_H2_AGENT", "weak")
    module = _import_script()

    with pytest.raises(SystemExit):
        module._real_mode_requested()

    captured = capsys.readouterr()
    assert "LIVE_H2_AGENT=weak" in captured.err
    assert "airline" in captured.err


def test_real_mode_requested_still_permits_weak_plus_retail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Regression guard: the relaxed guard must not stop permitting the
    # existing weak+retail path.
    monkeypatch.setenv("RUN_LIVE_H2_E2E", "1")
    monkeypatch.setenv("FIREWORKS_API_KEY", "fake-key-hermetic-test-only")
    monkeypatch.setenv("LIVE_H2_DOMAIN", "retail")
    monkeypatch.setenv("LIVE_H2_AGENT", "weak")
    module = _import_script()

    assert module._real_mode_requested() is True


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


# --- voice-sweep: bounded retry / error-record (no unbounded Retry-After) ---


def _probe(family: str = "retail-cancel_without_auth", text: str = "cancel W1") -> Any:
    from auditk.adapters.protocols import Stimulus
    from auditk.schema import ExpectedBehavior, ProbeDefinition

    ProbeDefinition.model_rebuild()
    return ProbeDefinition(
        probe_id="p-1",
        family=family,
        version="0.1",
        kind="prompt_injection",
        stimulus=Stimulus(channel="user", payload={"text": text}),
        expected_behavior=ExpectedBehavior(should_refuse=True),
    )


def _api_error() -> Exception:
    import httpx
    from openai import APITimeoutError

    return APITimeoutError(request=httpx.Request("POST", "http://x"))


def test_replay_with_retry_recovers_after_a_transient_api_error() -> None:
    from bossyk_sandbox.conditions.live_replay import LiveRunResult

    module = _import_script()
    calls = {"n": 0}

    def run_session(_payload: str) -> LiveRunResult:
        calls["n"] += 1
        if calls["n"] == 1:
            raise _api_error()
        return LiveRunResult(proposed=[], executed=[])

    slept: list[float] = []
    replay = module.replay_with_retry(_probe(), run_session, sleeper=slept.append)

    assert replay is not None
    assert calls["n"] == 2  # retried once
    assert slept  # backed off once (bounded, ignores Retry-After)


def test_replay_with_retry_returns_none_after_max_tries_never_hangs() -> None:
    module = _import_script()

    def run_session(_payload: str) -> Any:
        raise _api_error()

    slept: list[float] = []
    replay = module.replay_with_retry(
        _probe(), run_session, max_tries=3, backoffs_s=(0.0, 0.0, 0.0), sleeper=slept.append
    )

    assert replay is None  # recorded as errored, sweep continues
    assert len(slept) == 2  # slept between the 3 tries, not after the last


# --- agent banner (voice-model sweep: the banner must reflect AGENT_*, not a
# stale hardcoded label) -----------------------------------------------------


def test_agent_banner_reflects_the_resolved_agent_env_not_a_hardcoded_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The banner used to hardcode "Fireworks kimi-k2p6" regardless of which
    # provider AGENT_* actually pointed at (misleading once the sweep moved
    # to self-hosted RunPod models) -- it must print the REAL resolved
    # model @ base_url, from the same _resolve_agent_config seam
    # runtime.langgraph_agent uses to build the live agent.
    module = _import_script()
    monkeypatch.setenv("AGENT_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    monkeypatch.setenv("AGENT_API_KEY", "super-secret-runpod-key")
    monkeypatch.setenv("AGENT_BASE_URL", "https://api.runpod.ai/v2/wkdqe0qef23jy2/openai/v1")

    banner = module._agent_banner()

    assert (
        banner
        == "agent: Qwen/Qwen2.5-7B-Instruct @ https://api.runpod.ai/v2/wkdqe0qef23jy2/openai/v1"
    )
    assert "super-secret-runpod-key" not in banner  # the key is NEVER printed


def test_agent_banner_falls_back_to_fireworks_defaults_when_agent_env_is_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from bossyk_sandbox.runtime.langgraph_agent import FIREWORKS_BASE_URL

    module = _import_script()
    for var in ("AGENT_MODEL", "AGENT_API_KEY", "AGENT_BASE_URL", "FIREWORKS_MODEL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("FIREWORKS_API_KEY", "fw-key")

    banner = module._agent_banner()

    assert banner.startswith("agent: ")
    assert FIREWORKS_BASE_URL in banner
    assert "fw-key" not in banner
