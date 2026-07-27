#!/usr/bin/env python
"""Live H2/H4 real run (phase-live-h2h4-on-crossings.md, L1): replays each
frozen cooperative-adversary crossing in `probes/regression/<domain>.json`
through the REAL live LangGraph agent (Fireworks kimi-k2p6,
`conditions.live_replay.run_live_{airline,retail}_session`), scores it
post-hoc against the real policy judge (bossyk's `PolicyAwareJudge`,
`instruments.policy.build_default_policy_instrument`), and reports live-H2
(reach-boundary + agent-layer catch rate, per class/boundary, with policy
availability) and live-H4 (prevented vs detected-too-late + harm delta)
numbers, plus §15B latency and the judge token ledger.

Domain is selectable via `LIVE_H2_DOMAIN` (default `retail` -- the
retail-primary decision: the first billable L1 run targets retail, not
airline, per the locked design decision in
phase-live-h2h4-on-crossings.md). Drift is NOT scored (L1 is gate + policy
only -- see `conditions.live_boundary`'s module docstring for why).

Gated on FIREWORKS_API_KEY + RUN_LIVE_H2_E2E=1 (same shape as
scripts/benchmark_run.py / scripts/live_demo.py) -- real, billable API
traffic (agent calls + judge calls for every crossing in the corpus), never
run as part of the deterministic test suite. A smoke test
(tests/unit/test_live_h2h4_bench_script.py) imports this module and checks
it defines `main` -- it does NOT call `main()`.

Usage:
    RUN_LIVE_H2_E2E=1 FIREWORKS_API_KEY=... \\
        uv run python scripts/live_h2h4_bench.py                  # retail (default)
    RUN_LIVE_H2_E2E=1 LIVE_H2_DOMAIN=airline FIREWORKS_API_KEY=... \\
        uv run python scripts/live_h2h4_bench.py                  # airline
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from auditk.schema import ProbeDefinition
from openai import APIError

from bossyk_sandbox.conditions.adversary import TokenUsage
from bossyk_sandbox.conditions.live_multiturn import run_live_multiturn_retail_session
from bossyk_sandbox.conditions.live_replay import (
    CrossingReplay,
    LiveSessionResult,
    replay_crossing,
    run_live_airline_session,
    run_live_retail_session,
    run_live_weakened_retail_session,
    score_policy_post_hoc,
)
from bossyk_sandbox.conditions.retention import load_regression_probes
from bossyk_sandbox.domains import domain_config
from bossyk_sandbox.env import load_project_env
from bossyk_sandbox.instruments.policy import build_default_policy_instrument
from bossyk_sandbox.runtime.langgraph_agent import _resolve_agent_config
from bossyk_sandbox.scoring.cost import JudgeCallRecord, JudgeLedgerEntry, build_judge_token_ledger
from bossyk_sandbox.scoring.latency import LatencyRecord, LatencySummary, summarize_latency
from bossyk_sandbox.scoring.latency_budget import (
    DEFAULT_UX_BUDGETS_S,
    BudgetComparison,
    latency_budget,
)
from bossyk_sandbox.scoring.live_h2 import (
    CrossingScore,
    GroupSummary,
    RateWithCI,
    by_attack_class,
    by_boundary,
    live_h4,
    score_crossing,
)
from bossyk_sandbox.scoring.orthogonality import wilson_interval

REPO_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = REPO_ROOT / "docs" / "bench_output"
REGRESSION_PROBES_DIR = REPO_ROOT / "probes" / "regression"

# retail-primary (locked design decision): the first billable L1 run
# targets retail, not airline.
DOMAIN = os.environ.get("LIVE_H2_DOMAIN", "retail")

LIVE_H2_MODE = os.environ.get("LIVE_H2_MODE", "single")  # "single" (path A) | "multiturn" (path B)
LIVE_H2_MAX_TURNS = int(os.environ.get("LIVE_H2_MAX_TURNS", "8"))
LIVE_H2_AGENT = os.environ.get("LIVE_H2_AGENT", "compliant")  # "compliant" | "weak"

_RUN_SESSION_BY_DOMAIN: dict[str, Callable[[str], LiveSessionResult]] = {
    "airline": run_live_airline_session,
    "retail": run_live_retail_session,
}


def _multiturn_retail_session(payload: str) -> LiveSessionResult:
    # Path B: the corpus probe's payload is the adversarial GOAL; replay it as a
    # full multi-turn episode (retail-only).
    return run_live_multiturn_retail_session(payload, max_turns=LIVE_H2_MAX_TURNS)


def _run_session_for(domain: str) -> Callable[[str], LiveSessionResult]:
    if LIVE_H2_MODE == "multiturn":
        return _multiturn_retail_session
    if LIVE_H2_AGENT == "weak":
        return run_live_weakened_retail_session
    return _RUN_SESSION_BY_DOMAIN[domain]


def _corpus_path(domain: str) -> Path:
    # LIVE_H2_CORPUS points the bench at an alternate corpus (e.g. the
    # grounded corpus probes/grounded/<domain>.json) without a code change;
    # default is the per-domain H1 regression artifact.
    override = os.environ.get("LIVE_H2_CORPUS")
    if override:
        return Path(override)
    return REGRESSION_PROBES_DIR / f"{domain}.json"


def _output_path(domain: str) -> Path:
    # LIVE_H2_OUTPUT redirects the results file so a grounded (or any
    # alternate-corpus) run does not overwrite the committed per-domain L1
    # artifact; default is the per-domain bench output.
    override = os.environ.get("LIVE_H2_OUTPUT")
    if override:
        return Path(override)
    return OUTPUT_DIR / f"live_h2h4_{domain}.json"


def _real_mode_requested() -> bool:
    if os.environ.get("RUN_LIVE_H2_E2E") != "1":
        print("Set RUN_LIVE_H2_E2E=1 to run the live H2/H4 real run.", file=sys.stderr)
        raise SystemExit(1)
    if not os.environ.get("FIREWORKS_API_KEY"):
        print("FIREWORKS_API_KEY is required for the live H2/H4 real run.", file=sys.stderr)
        raise SystemExit(1)
    # DOMAIN must be registered in BOTH the scenario/policy registry
    # (domains.domain_config, raises KeyError below in main() if not) AND
    # the live-session-runner registry (L2 domains, if any, may lag behind
    # domain_config -- this is a distinct failure mode worth its own
    # message rather than a bare KeyError from a dict lookup).
    if DOMAIN not in _RUN_SESSION_BY_DOMAIN:
        print(
            f"LIVE_H2_DOMAIN={DOMAIN!r} has no live session runner "
            f"(registered: {sorted(_RUN_SESSION_BY_DOMAIN)}).",
            file=sys.stderr,
        )
        raise SystemExit(1)
    if LIVE_H2_MODE not in {"single", "multiturn"}:
        print(f"LIVE_H2_MODE={LIVE_H2_MODE!r} must be 'single' or 'multiturn'.", file=sys.stderr)
        raise SystemExit(1)
    if LIVE_H2_MODE == "multiturn" and DOMAIN != "retail":
        print(f"LIVE_H2_MODE=multiturn only supports retail (got {DOMAIN!r}).", file=sys.stderr)
        raise SystemExit(1)
    if LIVE_H2_AGENT not in {"compliant", "weak"}:
        print(f"LIVE_H2_AGENT={LIVE_H2_AGENT!r} must be 'compliant' or 'weak'.", file=sys.stderr)
        raise SystemExit(1)
    if LIVE_H2_AGENT == "weak" and DOMAIN != "retail":
        print(f"LIVE_H2_AGENT=weak only supports retail (got {DOMAIN!r}).", file=sys.stderr)
        raise SystemExit(1)
    return True


def _agent_banner() -> str:
    """The bench's agent banner, resolved from the SAME `AGENT_*`/
    `FIREWORKS_*` env seam `_build_agent_session` uses
    (`runtime.langgraph_agent._resolve_agent_config`) -- so it reflects
    whichever provider/model is actually driving the run (a self-hosted
    RunPod model, Fireworks, ...) instead of a hardcoded, stale label
    (it used to always print "Fireworks kimi-k2p6" regardless of AGENT_*).
    Prints `model @ base_url`; NEVER the key."""
    model, _api_key, base_url = _resolve_agent_config(model_name=None, api_key=None, base_url=None)
    return f"agent: {model} @ {base_url}"


def _rate_to_dict(rate: RateWithCI) -> dict[str, object]:
    return asdict(rate)


def _group_summary_to_dict(summary: GroupSummary) -> dict[str, object]:
    return {
        "group": summary.group,
        "n_crossings": summary.n_crossings,
        "reach": _rate_to_dict(summary.reach),
        "catch": _rate_to_dict(summary.catch),
        "n_policy_scored": summary.n_policy_scored,
        "n_policy_error": summary.n_policy_error,
        "n_policy_unscored": summary.n_policy_unscored,
        "n_policy_missing": summary.n_policy_missing,
    }


def _score_to_dict(score: CrossingScore) -> dict[str, object]:
    return {
        "probe_id": score.probe_id,
        "domain": score.domain,
        "boundary": score.boundary,
        "attack_class": score.attack_class,
        "reached": score.reached,
        "prevented": score.prevented,
        "detected": score.detected,
        "caught": score.caught,
        "detected_too_late": score.detected_too_late,
    }


def _judge_ledger_to_dict(ledger: dict[str, JudgeLedgerEntry]) -> dict[str, object]:
    return {
        instrument: {
            "calls": entry.calls,
            "error_calls": entry.error_calls,
            "input_tokens": entry.usage.input_tokens,
            "output_tokens": entry.usage.output_tokens,
            "total_tokens": entry.usage.total_tokens,
        }
        for instrument, entry in ledger.items()
    }


def _latency_summary_to_dict(summaries: dict[str, LatencySummary]) -> dict[str, object]:
    return {instrument: asdict(summary) for instrument, summary in summaries.items()}


def _mean_action_exec_s(records: list[LatencyRecord]) -> float | None:
    """Mean wall-clock of the tau2 tool calls the run actually executed --
    the measured action-exec budget floor. `None` when no action executed
    (every proposed action gate-blocked pre-execution), so the latency budget
    reports the UX sweep alone rather than a fabricated floor."""
    if not records:
        return None
    return sum(record.elapsed_s for record in records) / len(records)


def _latency_budget_to_dict(
    budget: dict[str, list[BudgetComparison]],
) -> dict[str, list[dict[str, object]]]:
    """Serialize the per-detector speedup targets (the §15C number to beat)
    for the results JSON. Deterministic recompute from the `latency` block --
    detection latency vs each hold budget."""
    return {
        instrument: [asdict(comparison) for comparison in comparisons]
        for instrument, comparisons in budget.items()
    }


def _print_group_summaries(label: str, groups: dict[str, GroupSummary]) -> None:
    print(f"\n-- live-H2 by {label} --")
    for key in sorted(groups):
        summary = groups[key]
        reach, catch = summary.reach, summary.catch
        print(
            f"  {key}: reach={reach.successes}/{reach.n}={reach.rate:.3f} "
            f"[{reach.ci_low:.3f}, {reach.ci_high:.3f}]  "
            f"catch={catch.successes}/{catch.n}={catch.rate:.3f} "
            f"[{catch.ci_low:.3f}, {catch.ci_high:.3f}]  "
            f"(policy: scored={summary.n_policy_scored} error={summary.n_policy_error} "
            f"unscored={summary.n_policy_unscored} missing={summary.n_policy_missing})"
        )


def replay_with_retry(
    probe: ProbeDefinition,
    run_session: Callable[[str], LiveSessionResult],
    *,
    max_tries: int = 3,
    backoffs_s: tuple[float, ...] = (5.0, 10.0, 20.0),
    sleeper: Callable[[float], None] = time.sleep,
) -> CrossingReplay | None:
    """Replay one crossing with BOUNDED retry on provider API errors (rate
    limits, timeouts, connection). Backoff is capped and IGNORES any
    `Retry-After` -- the sweep must never sleep for a provider's quota window
    (an unbounded client-side Retry-After hung a run ~10h). Returns None if the
    attempt still fails after `max_tries`, so the caller records it as an error
    and the sweep continues instead of crashing."""
    for attempt in range(max_tries):
        try:
            return replay_crossing(probe, run_session)
        except APIError as exc:
            if attempt == max_tries - 1:
                print(f"    ERRORED after {max_tries} tries: {type(exc).__name__}: {str(exc)[:80]}")
                return None
            sleeper(backoffs_s[min(attempt, len(backoffs_s) - 1)])
    return None


def main() -> None:
    # Load .env before the gate check (non-overriding -- an exported value
    # in the shell always wins).
    load_project_env()
    _real_mode_requested()
    run_session = _run_session_for(DOMAIN)
    corpus_path = _corpus_path(DOMAIN)
    cfg = domain_config(DOMAIN)

    print(f"=== Live H2/H4 real run (domain={DOMAIN}) ===")
    print(f"corpus: {corpus_path}")
    print(f"policy: {cfg.policy_path}")
    print(_agent_banner())
    print("drift: SKIPPED (L1 scope -- see conditions/live_boundary.py)")
    print()

    probes: list[ProbeDefinition] = load_regression_probes(corpus_path)
    print(f"Loaded {len(probes)} frozen crossings.\n")

    usage_records: list[JudgeCallRecord] = []

    def _on_policy_call(usage: TokenUsage, errored: bool) -> None:
        usage_records.append(JudgeCallRecord(instrument="policy", usage=usage, errored=errored))

    # Pace between attempts to stay under the provider's per-minute limit
    # proactively (env-tunable; 0 disables). Bounded retry (above) is the
    # backstop, pacing is the avoidance.
    inter_attempt_delay_s = float(os.environ.get("LIVE_H2_PACE_S", "2"))

    scores: list[CrossingScore] = []
    all_latency: list[LatencyRecord] = []
    all_action_latency: list[LatencyRecord] = []
    all_agent_latency: list[LatencyRecord] = []
    n_engaged = 0
    n_errored = 0
    for index, probe in enumerate(probes, start=1):
        print(f"[{index}/{len(probes)}] replaying {probe.probe_id} ...")
        if index > 1 and inter_attempt_delay_s > 0:
            time.sleep(inter_attempt_delay_s)
        replay = replay_with_retry(probe, run_session)
        if replay is None:
            # Persistent provider error -- recorded, excluded from the rate
            # denominators (like the judge error accounting), sweep continues.
            n_errored += 1
            continue
        all_action_latency.extend(replay.tool_latency)
        all_agent_latency.extend(replay.agent_latency)
        # ENGAGEMENT: the agent proposed >=1 tool call this attempt. The
        # voice-model sweep reads this to tell a robust 0-reach (engaged but
        # policy-compliant) from an incapable one (never drove the tools).
        if replay.proposed:
            n_engaged += 1

        policy = build_default_policy_instrument(cfg.policy_path)
        policy.on_call = _on_policy_call
        scored = score_policy_post_hoc(replay, policy)
        all_latency.extend(scored.latency)

        score = score_crossing(replay, scored.verdicts)
        scores.append(score)
        print(
            f"    reached={score.reached} prevented={score.prevented} "
            f"detected={score.detected} caught={score.caught}"
        )

    judge_ledger = build_judge_token_ledger(usage_records)
    # Combine policy (judge) + agent-inference records so the `latency` block
    # carries both -- agent_inference is the voice-viability metric.
    latency_summary = summarize_latency(all_latency + all_agent_latency)
    action_exec_summary = summarize_latency(all_action_latency)
    # Denominators over SUCCESSFUL attempts (errored attempts excluded, counted
    # separately in n_errored) so rates aren't diluted by provider failures.
    n_scored = len(scores)
    engaged_low, engaged_high = wilson_interval(n_engaged, n_scored) if n_scored else (0.0, 0.0)
    engagement = RateWithCI(
        n=n_scored,
        successes=n_engaged,
        rate=(n_engaged / n_scored if n_scored else 0.0),
        ci_low=engaged_low,
        ci_high=engaged_high,
    )
    action_exec_s = _mean_action_exec_s(all_action_latency)
    budget = latency_budget(
        latency_summary, action_exec_s=action_exec_s, ux_budgets_s=DEFAULT_UX_BUDGETS_S
    )

    boundary_groups = by_boundary(scores)
    class_groups = by_attack_class(scores)
    h4 = live_h4(scores)

    _print_group_summaries("boundary", boundary_groups)
    _print_group_summaries("attack class", class_groups)

    print("\n-- engagement (agent proposed >=1 tool call) --")
    print(
        f"  engaged={engagement.successes}/{engagement.n}={engagement.rate:.3f} "
        f"[{engagement.ci_low:.3f}, {engagement.ci_high:.3f}]  "
        f"(errored attempts excluded: {n_errored}/{len(probes)}) "
        "(0-reach is only robustness if engagement is high)"
    )

    print("\n-- live-H4 (prevented vs detected-too-late) --")
    print(f"n_violations={h4.n_violations}")
    print(f"prevented={h4.prevented}")
    print(f"detected_too_late={h4.detected_too_late}")
    print(f"undetected={h4.undetected}")
    print(f"harm_off={h4.harm_off} harm_on={h4.harm_on} harm_delta={h4.harm_delta}")

    print("\n-- judge token ledger --")
    for instrument, entry in judge_ledger.items():
        print(
            f"  {instrument}: {entry.calls} calls ({entry.error_calls} errored), "
            f"total_tokens={entry.usage.total_tokens}"
        )

    print("\n-- latency (wall-clock, seconds) --")
    for instrument, summary in latency_summary.items():
        print(
            f"  {instrument}: n={summary.count} mean={summary.mean_s:.3f} "
            f"p50={summary.p50_s:.3f} p95={summary.p95_s:.3f} max={summary.max_s:.3f}"
        )

    print("\n-- latency budget (§15B+: detection latency vs hold budget) --")
    if action_exec_s is None:
        print("  action-exec floor: n/a (no tool executed -- all actions gate-blocked)")
    else:
        print(
            f"  action-exec floor: {action_exec_s * 1000:.3f}ms "
            "(mean tau2 tool-call cost -- an in-process floor, not real-world action I/O)"
        )
    for instrument, comparisons in budget.items():
        for comparison in comparisons:
            print(
                f"  {instrument} vs {comparison.budget_label} "
                f"({comparison.budget_s:.3f}s): needs "
                f"{comparison.speedup_needed_mean:.1f}x (mean) / "
                f"{comparison.speedup_needed_p95:.1f}x (p95) faster; "
                f"prevented_in_time={comparison.prevented_in_time_at_mean}"
            )

    output = {
        "generated_at": datetime.now(UTC).isoformat(),
        "domain": DOMAIN,
        "corpus": str(corpus_path),
        "n_crossings": len(scores),
        "n_attempts": len(probes),
        "n_errored": n_errored,
        "engagement": _rate_to_dict(engagement),
        "crossings": [_score_to_dict(score) for score in scores],
        "live_h2_by_boundary": {
            key: _group_summary_to_dict(value) for key, value in boundary_groups.items()
        },
        "live_h2_by_attack_class": {
            key: _group_summary_to_dict(value) for key, value in class_groups.items()
        },
        "live_h4": asdict(h4),
        "judge_token_ledger": _judge_ledger_to_dict(judge_ledger),
        "latency": _latency_summary_to_dict(latency_summary),
        "action_exec_latency": _latency_summary_to_dict(action_exec_summary),
        "latency_budget": _latency_budget_to_dict(budget),
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = _output_path(DOMAIN)
    output_path.write_text(json.dumps(output, indent=2))
    print(f"\nWrote results to {output_path}")


if __name__ == "__main__":
    main()
