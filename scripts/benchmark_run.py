#!/usr/bin/env python
"""Phase 2c cross-domain benchmark run: real drift + policy judges over
each registered domain's scripted scenario set (see
`bossyk_sandbox.domains`), producing a combined 3-way orthogonality table
across all domains plus a per-domain breakdown, and the gate-vs-ground-truth
confusion read (B2 safety-weighted + B3 bind/no-bind) both combined and
per-domain. Also persists full per-step output (declared_intent, action,
both verdicts + reasoning, tagged with domain) to docs/bench_output/ so
future runs and diagnostics read from disk instead of a recompute.

Which domains run is controlled by the `BENCH_DOMAINS` env var (comma
separated domain names, default "airline,retail") -- each name is looked up
via `bossyk_sandbox.domains.domain_config`, so only registered domains can
be selected.

Gated on FIREWORKS_API_KEY + RUN_SANDBOX_BENCH=1 (same shape as Phase 0's
scripts/live_demo.py) -- not part of the test suite, run explicitly:

    uv sync --extra bench   # pulls in auditk[nli] (torch/transformers)
    RUN_SANDBOX_BENCH=1 RUN_JUDGE_MODEL=1 RUN_NLI_MODEL=1 \\
        uv run python scripts/benchmark_run.py
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from bossyk_sandbox.domains import domain_config
from bossyk_sandbox.env import load_project_env
from bossyk_sandbox.instruments.base import Verdict
from bossyk_sandbox.instruments.drift import build_default_drift_instrument
from bossyk_sandbox.instruments.outcome_key import OutcomeKeyLookup
from bossyk_sandbox.instruments.policy import build_default_policy_instrument
from bossyk_sandbox.scenarios.loader import load_scenarios, outcome_keys
from bossyk_sandbox.scenarios.runner import (
    FIRING_LABELS,
    InstrumentAvailability,
    ScoredStep,
    instrument_availability,
    run_scenario,
    to_gate_outcomes,
    to_membership,
)
from bossyk_sandbox.scoring.confusion import (
    ConfusionMatrix,
    GateOutcomeRecord,
    b2_safety_weighted,
    b3_bind_headline,
    binary_confusion,
)
from bossyk_sandbox.scoring.interrupt import H4Result, InterruptRecord, h4_result
from bossyk_sandbox.scoring.orthogonality import StepMembership, orthogonality_table, split_complete

OUTPUT_DIR = Path(__file__).parent.parent / "docs" / "bench_output"
DEFAULT_BENCH_DOMAINS = "airline,retail"


def _bench_domains() -> list[str]:
    """Parses `BENCH_DOMAINS` (comma separated, default "airline,retail")
    into the ordered list of domain names to run this benchmark over."""
    raw = os.environ.get("BENCH_DOMAINS", DEFAULT_BENCH_DOMAINS)
    return [name.strip() for name in raw.split(",") if name.strip()]


def _judge_availability(all_scored_steps: list[ScoredStep]) -> dict[str, InstrumentAvailability]:
    """Per-instrument availability (Finding 4) for the drift and policy
    judges: how many steps were actually scored versus errored, unscored,
    or never annotated at all."""
    return {
        "drift": instrument_availability(all_scored_steps, "drift"),
        "policy": instrument_availability(all_scored_steps, "policy"),
    }


def _report_judge_errors(all_scored_steps: list[ScoredStep]) -> None:
    """Judge calls (real Fireworks traffic) can fail (see
    instruments/policy.py, instruments/drift.py) -- surface how many steps
    got an "error" verdict instead of silently folding them into
    "non-firing", which would understate the true rates."""
    total = len(all_scored_steps)
    availability = _judge_availability(all_scored_steps)
    print(f"=== Judge availability (of {total} scored steps) ===")
    for instrument_name, avail in availability.items():
        print(
            f"{instrument_name}: n_scored={avail.n_scored} n_error={avail.n_error} "
            f"n_unscored={avail.n_unscored} n_missing={avail.n_missing}"
        )
    print()


def _print_orthogonality_table(memberships: list[StepMembership]) -> None:
    complete, incomplete = split_complete(memberships)
    print(
        f"(n_membership_complete={len(complete)} "
        f"n_membership_dropped_unavailable={len(incomplete)})"
    )
    for cell in orthogonality_table(complete):
        print(
            f"drift={cell.cell.drift_fires} policy={cell.cell.policy_fires} "
            f"outcome_violation={cell.cell.outcome_violation}: "
            f"n={cell.count} p={cell.proportion:.3f} "
            f"CI=[{cell.ci_low:.3f}, {cell.ci_high:.3f}]"
        )


def _print_confusion_read(label: str, matrix: ConfusionMatrix) -> None:
    safety = b2_safety_weighted(matrix)
    headline = b3_bind_headline(matrix)
    print(f"\n=== {label}: B2 safety-weighted confusion (false-admit vs false-hold) ===")
    print(f"false_admit={safety.false_admit} (rate={safety.false_admit_rate:.3f})")
    print(f"false_hold={safety.false_hold} (rate={safety.false_hold_rate:.3f})")
    print(f"\n=== {label}: B3 bind/no-bind headline ===")
    print(f"accuracy={headline.accuracy:.3f}")
    print(f"bind_precision={headline.bind_precision:.3f}")
    print(f"bind_recall={headline.bind_recall:.3f}")


def _domain_report(
    memberships: list[StepMembership], gate_outcomes: list[GateOutcomeRecord]
) -> dict[str, object]:
    matrix = binary_confusion(gate_outcomes)
    complete, incomplete = split_complete(memberships)
    return {
        "orthogonality_table": [asdict(cell) for cell in orthogonality_table(complete)],
        "n_membership_complete": len(complete),
        "n_membership_dropped_unavailable": len(incomplete),
        "confusion_matrix": asdict(matrix),
        "safety_weighted": asdict(b2_safety_weighted(matrix)),
        "bind_headline": asdict(b3_bind_headline(matrix)),
    }


def _to_interrupt_records(
    scored_steps: list[ScoredStep], outcome_lookup: OutcomeKeyLookup
) -> list[InterruptRecord]:
    """Joins scored steps against the A1 ground truth to build the H4
    interrupt-efficacy input (SCOUT.md Phase 3): was each ground-truth
    violation prevented pre-execution by the gate, caught too late by a
    slow instrument (drift or policy firing post-hoc), or missed entirely.
    Mirrors to_membership/to_gate_outcomes in scenarios/runner.py."""
    records: list[InterruptRecord] = []
    for scored in scored_steps:
        is_violation = outcome_lookup.is_violation(scored.scenario_id, scored.step_index)
        if is_violation is None:
            continue
        drift_v = next((v for v in scored.verdicts if v.instrument == "drift"), None)
        policy_v = next((v for v in scored.verdicts if v.instrument == "policy"), None)
        # An "error"/"unscored" verdict is deliberately treated the same as a
        # non-firing one here (never in FIRING_LABELS) -- an instrument that
        # errored cannot be credited with a detection. This can only
        # understate H4's detected_too_late/prevention read (conservative),
        # never overstate it; instrument availability (Finding 4) is
        # reported separately via `instrument_availability` so an outage is
        # visible on its own, not laundered into this join.
        slow_detected = (drift_v is not None and drift_v.label in FIRING_LABELS) or (
            policy_v is not None and policy_v.label in FIRING_LABELS
        )
        records.append(
            InterruptRecord(
                is_violation=is_violation,
                gate_blocked=scored.decision.verdict is Verdict.BLOCK,
                slow_detected=slow_detected,
            )
        )
    return records


def _print_h4_read(label: str, result: H4Result) -> None:
    low, high = result.prevention_ci95()
    print(f"\n=== {label}: H4 interrupt efficacy ===")
    print(f"n_violations={result.n_violations}")
    print(f"prevented={result.prevented}")
    print(f"detected_too_late={result.detected_too_late}")
    print(f"undetected={result.undetected}")
    print(f"harm_off={result.harm_off}")
    print(f"harm_on={result.harm_on}")
    print(f"harm_delta={result.harm_delta}")
    print(f"prevention_rate={result.prevention_rate():.3f} CI=[{low:.3f}, {high:.3f}]")


def _h4_report(result: H4Result) -> dict[str, object]:
    low, high = result.prevention_ci95()
    return {
        **asdict(result),
        "prevention_rate": result.prevention_rate(),
        "prevention_ci_low": low,
        "prevention_ci_high": high,
    }


def _per_step_rows(
    scored_steps_by_domain: dict[str, list[ScoredStep]],
    lookups_by_domain: dict[str, OutcomeKeyLookup],
) -> list[dict[str, object]]:
    """Full per-step judge output -- scenario/step, boundary_label,
    declared_intent, action, both verdicts + reasoning, tagged with domain
    -- so a diagnostic can inspect exactly what a benchmark run computed
    without recomputing it (see docs/drift-diagnostic-findings.md, which
    had to re-run the whole benchmark because this didn't exist yet)."""
    rows: list[dict[str, object]] = []
    for domain_name, scored_steps in scored_steps_by_domain.items():
        lookup = lookups_by_domain[domain_name]
        for scored in scored_steps:
            boundary = lookup.label_for(scored.scenario_id, scored.step_index)
            drift_v = next((v for v in scored.verdicts if v.instrument == "drift"), None)
            policy_v = next((v for v in scored.verdicts if v.instrument == "policy"), None)
            rows.append(
                {
                    "domain": domain_name,
                    "scenario_id": scored.scenario_id,
                    "step_index": scored.step_index,
                    "boundary_label": boundary.value if boundary else None,
                    "outcome_violation": lookup.is_violation(scored.scenario_id, scored.step_index),
                    "tool_name": scored.step.action.payload.get("tool_name"),
                    "arguments": scored.step.action.payload.get("arguments"),
                    "declared_intent": scored.step.declared_intent,
                    "gate_verdict": scored.decision.verdict.value,
                    "gate_reason": scored.decision.reason,
                    "drift": asdict(drift_v) if drift_v else None,
                    "policy": asdict(policy_v) if policy_v else None,
                }
            )
    return rows


def _write_report(
    *,
    domain_names: list[str],
    scored_steps_by_domain: dict[str, list[ScoredStep]],
    lookups_by_domain: dict[str, OutcomeKeyLookup],
    memberships_by_domain: dict[str, list[StepMembership]],
    gate_outcomes_by_domain: dict[str, list[GateOutcomeRecord]],
    interrupt_records_by_domain: dict[str, list[InterruptRecord]],
    all_memberships: list[StepMembership],
    all_gate_outcomes: list[GateOutcomeRecord],
    all_interrupt_records: list[InterruptRecord],
) -> Path:
    """Persists the combined + per-domain 3-way orthogonality tables,
    confusion reads, and H4 interrupt-efficacy reads, plus full per-step
    judge output tagged with domain -- the phase2c cross-domain extension
    of phase1's single-domain report."""
    all_scored_steps = [step for steps in scored_steps_by_domain.values() for step in steps]
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "combined": _domain_report(all_memberships, all_gate_outcomes),
        "per_domain": {
            domain_name: _domain_report(
                memberships_by_domain[domain_name], gate_outcomes_by_domain[domain_name]
            )
            for domain_name in domain_names
        },
        "h4": {
            "combined": _h4_report(h4_result(all_interrupt_records)),
            "per_domain": {
                domain_name: _h4_report(h4_result(interrupt_records_by_domain[domain_name]))
                for domain_name in domain_names
            },
        },
        "availability": {
            "combined": {
                instrument: asdict(avail)
                for instrument, avail in _judge_availability(all_scored_steps).items()
            },
            "per_domain": {
                domain_name: {
                    instrument: asdict(avail)
                    for instrument, avail in _judge_availability(
                        scored_steps_by_domain[domain_name]
                    ).items()
                }
                for domain_name in domain_names
            },
        },
        "steps": _per_step_rows(scored_steps_by_domain, lookups_by_domain),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "phase2c_orthogonality.json"
    out_path.write_text(json.dumps(report, indent=2))
    return out_path


def main() -> None:
    # Load .env before the key gates below (non-overriding: shell wins).
    load_project_env()
    if os.environ.get("RUN_SANDBOX_BENCH") != "1":
        print("Set RUN_SANDBOX_BENCH=1 to run the real-judge benchmark.", file=sys.stderr)
        raise SystemExit(1)
    if not os.environ.get("FIREWORKS_API_KEY"):
        print("FIREWORKS_API_KEY is required for the real-judge benchmark.", file=sys.stderr)
        raise SystemExit(1)

    domain_names = _bench_domains()

    # Drift is domain-agnostic (it scores declared-intent-vs-action text,
    # not domain-specific policy) so it's built once and reused; the policy
    # instrument is loaded against a specific policy_path per domain, so it
    # must be built per domain.
    drift_instrument = build_default_drift_instrument()

    scored_steps_by_domain: dict[str, list[ScoredStep]] = {}
    lookups_by_domain: dict[str, OutcomeKeyLookup] = {}
    memberships_by_domain: dict[str, list[StepMembership]] = {}
    gate_outcomes_by_domain: dict[str, list[GateOutcomeRecord]] = {}
    interrupt_records_by_domain: dict[str, list[InterruptRecord]] = {}

    for domain_name in domain_names:
        cfg = domain_config(domain_name)
        scenarios = load_scenarios(cfg.scenarios_path)
        lookup = OutcomeKeyLookup(keys=outcome_keys(scenarios))
        policy_instrument = build_default_policy_instrument(cfg.policy_path)

        domain_scored_steps: list[ScoredStep] = []
        for scenario in scenarios:
            _, scenario_scored_steps = run_scenario(
                scenario,
                slow_instruments=[drift_instrument, policy_instrument],
                fast_rules=cfg.fast_rules_factory(),
            )
            domain_scored_steps.extend(scenario_scored_steps)

        scored_steps_by_domain[domain_name] = domain_scored_steps
        lookups_by_domain[domain_name] = lookup
        memberships_by_domain[domain_name] = to_membership(domain_scored_steps, lookup)
        gate_outcomes_by_domain[domain_name] = to_gate_outcomes(domain_scored_steps, lookup)
        interrupt_records_by_domain[domain_name] = _to_interrupt_records(
            domain_scored_steps, lookup
        )

    all_scored_steps = [step for steps in scored_steps_by_domain.values() for step in steps]
    all_memberships = [m for ms in memberships_by_domain.values() for m in ms]
    all_gate_outcomes = [g for gs in gate_outcomes_by_domain.values() for g in gs]
    all_interrupt_records = [r for rs in interrupt_records_by_domain.values() for r in rs]

    out_path = _write_report(
        domain_names=domain_names,
        scored_steps_by_domain=scored_steps_by_domain,
        lookups_by_domain=lookups_by_domain,
        memberships_by_domain=memberships_by_domain,
        gate_outcomes_by_domain=gate_outcomes_by_domain,
        interrupt_records_by_domain=interrupt_records_by_domain,
        all_memberships=all_memberships,
        all_gate_outcomes=all_gate_outcomes,
        all_interrupt_records=all_interrupt_records,
    )
    print(
        f"Wrote {len(all_scored_steps)} steps across {len(domain_names)} domain(s) to {out_path}\n"
    )

    _report_judge_errors(all_scored_steps)

    print("=== Combined 3-way orthogonality table (drift / policy / outcome) ===")
    _print_orthogonality_table(all_memberships)
    for domain_name in domain_names:
        print(f"\n=== {domain_name}: 3-way orthogonality table (drift / policy / outcome) ===")
        _print_orthogonality_table(memberships_by_domain[domain_name])

    _print_confusion_read("Combined", binary_confusion(all_gate_outcomes))
    for domain_name in domain_names:
        _print_confusion_read(domain_name, binary_confusion(gate_outcomes_by_domain[domain_name]))

    _print_h4_read("Combined", h4_result(all_interrupt_records))
    for domain_name in domain_names:
        _print_h4_read(domain_name, h4_result(interrupt_records_by_domain[domain_name]))


if __name__ == "__main__":
    main()
