#!/usr/bin/env python
"""Phase 1 benchmark run: real drift + policy judges over the scripted
airline scenario set, producing the 3-way orthogonality table and the
gate-vs-ground-truth confusion read (B2 safety-weighted + B3 bind/no-bind).

Gated on FIREWORKS_API_KEY + RUN_SANDBOX_BENCH=1 (same shape as Phase 0's
scripts/live_demo.py) -- not part of the test suite, run explicitly:

    RUN_SANDBOX_BENCH=1 RUN_JUDGE_MODEL=1 RUN_NLI_MODEL=1 \\
        uv run python scripts/benchmark_run.py
"""

from __future__ import annotations

import os
import sys

from bossyk_sandbox.instruments.drift import build_default_drift_instrument
from bossyk_sandbox.instruments.outcome_key import OutcomeKeyLookup
from bossyk_sandbox.instruments.policy import build_default_policy_instrument
from bossyk_sandbox.scenarios.loader import load_scenarios, outcome_keys
from bossyk_sandbox.scenarios.runner import run_scenario, to_gate_outcomes, to_membership
from bossyk_sandbox.scoring.confusion import b2_safety_weighted, b3_bind_headline, binary_confusion
from bossyk_sandbox.scoring.orthogonality import orthogonality_table


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
    for scenario in scenarios:
        _, scored_steps = run_scenario(
            scenario, slow_instruments=[drift_instrument, policy_instrument]
        )
        all_memberships.extend(to_membership(scored_steps, lookup))
        all_gate_outcomes.extend(to_gate_outcomes(scored_steps, lookup))

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
