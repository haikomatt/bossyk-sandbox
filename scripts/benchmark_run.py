#!/usr/bin/env python
"""Phase 1 benchmark run: real drift + policy judges over the scripted
airline scenario set, producing the 3-way orthogonality table and the
gate-vs-ground-truth confusion read (B2 safety-weighted + B3 bind/no-bind).
Also persists full per-step output (declared_intent, action, both verdicts +
reasoning) to docs/bench_output/ so future runs and diagnostics read from
disk instead of a recompute.

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

from bossyk_sandbox.instruments.drift import ERROR_LABEL as DRIFT_ERROR_LABEL
from bossyk_sandbox.instruments.drift import build_default_drift_instrument
from bossyk_sandbox.instruments.outcome_key import OutcomeKeyLookup
from bossyk_sandbox.instruments.policy import ERROR_LABEL as POLICY_ERROR_LABEL
from bossyk_sandbox.instruments.policy import build_default_policy_instrument
from bossyk_sandbox.scenarios.loader import load_scenarios, outcome_keys
from bossyk_sandbox.scenarios.runner import (
    ScoredStep,
    run_scenario,
    to_gate_outcomes,
    to_membership,
)
from bossyk_sandbox.scoring.confusion import b2_safety_weighted, b3_bind_headline, binary_confusion
from bossyk_sandbox.scoring.orthogonality import orthogonality_table

OUTPUT_DIR = Path(__file__).parent.parent / "docs" / "bench_output"


def _report_judge_errors(all_scored_steps: list[ScoredStep]) -> None:
    """Judge calls (real Fireworks traffic) can fail (see
    instruments/policy.py, instruments/drift.py) -- surface how many steps
    got an "error" verdict instead of silently folding them into
    "non-firing", which would understate the true rates."""
    total = len(all_scored_steps)
    drift_errors = sum(
        1
        for s in all_scored_steps
        if any(v.instrument == "drift" and v.label == DRIFT_ERROR_LABEL for v in s.verdicts)
    )
    policy_errors = sum(
        1
        for s in all_scored_steps
        if any(v.instrument == "policy" and v.label == POLICY_ERROR_LABEL for v in s.verdicts)
    )
    print(f"=== Judge error rate (of {total} scored steps) ===")
    print(
        f"drift errors: {drift_errors} ({drift_errors / total:.1%})" if total else "drift errors: 0"
    )
    print(
        f"policy errors: {policy_errors} ({policy_errors / total:.1%})"
        if total
        else "policy errors: 0"
    )
    print()


def _persist_per_step_output(all_scored_steps: list[ScoredStep], lookup: OutcomeKeyLookup) -> Path:
    """Writes full per-step judge output to disk -- scenario/step,
    boundary_label, declared_intent, action, both verdicts + reasoning --
    so a diagnostic can inspect exactly what a benchmark run computed
    without recomputing it (see docs/drift-diagnostic-findings.md, which
    had to re-run the whole benchmark because this didn't exist yet)."""
    rows: list[dict[str, object]] = []
    for scored in all_scored_steps:
        boundary = lookup.label_for(scored.scenario_id, scored.step_index)
        drift_v = next((v for v in scored.verdicts if v.instrument == "drift"), None)
        policy_v = next((v for v in scored.verdicts if v.instrument == "policy"), None)
        rows.append(
            {
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

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "phase1_benchmark_run.json"
    out_path.write_text(
        json.dumps({"generated_at": datetime.now(UTC).isoformat(), "steps": rows}, indent=2)
    )
    return out_path


def main() -> None:
    if os.environ.get("RUN_SANDBOX_BENCH") != "1":
        print("Set RUN_SANDBOX_BENCH=1 to run the real-judge benchmark.", file=sys.stderr)
        raise SystemExit(1)
    if not os.environ.get("FIREWORKS_API_KEY"):
        print("FIREWORKS_API_KEY is required for the real-judge benchmark.", file=sys.stderr)
        raise SystemExit(1)

    scenarios = load_scenarios()
    lookup = OutcomeKeyLookup(keys=outcome_keys(scenarios))

    drift_instrument = build_default_drift_instrument()
    policy_instrument = build_default_policy_instrument()

    all_memberships = []
    all_gate_outcomes = []
    all_scored_steps: list[ScoredStep] = []
    for scenario in scenarios:
        _, scored_steps = run_scenario(
            scenario, slow_instruments=[drift_instrument, policy_instrument]
        )
        all_scored_steps.extend(scored_steps)
        all_memberships.extend(to_membership(scored_steps, lookup))
        all_gate_outcomes.extend(to_gate_outcomes(scored_steps, lookup))

    out_path = _persist_per_step_output(all_scored_steps, lookup)
    print(f"Wrote {len(all_scored_steps)} steps to {out_path}\n")

    _report_judge_errors(all_scored_steps)

    print("=== 3-way orthogonality table (drift / policy / outcome) ===")
    for cell in orthogonality_table(all_memberships):
        print(
            f"drift={cell.cell.drift_fires} policy={cell.cell.policy_fires} "
            f"outcome_violation={cell.cell.outcome_violation}: "
            f"n={cell.count} p={cell.proportion:.3f} "
            f"CI=[{cell.ci_low:.3f}, {cell.ci_high:.3f}]"
        )

    matrix = binary_confusion(all_gate_outcomes)
    safety = b2_safety_weighted(matrix)
    headline = b3_bind_headline(matrix)

    print("\n=== B2 safety-weighted confusion (false-admit vs false-hold) ===")
    print(f"false_admit={safety.false_admit} (rate={safety.false_admit_rate:.3f})")
    print(f"false_hold={safety.false_hold} (rate={safety.false_hold_rate:.3f})")

    print("\n=== B3 bind/no-bind headline ===")
    print(f"accuracy={headline.accuracy:.3f}")
    print(f"bind_precision={headline.bind_precision:.3f}")
    print(f"bind_recall={headline.bind_recall:.3f}")


if __name__ == "__main__":
    main()
