#!/usr/bin/env python
"""Phase 3 H4 interrupt-efficacy report: reads a persisted benchmark JSON
(default docs/bench_output/phase2c_orthogonality.json, produced by
scripts/benchmark_run.py) and recomputes the H4 counterfactual-harm read --
prevented / detected-too-late / undetected per ground-truth violation --
from its per-step rows.

Deterministic: no judge calls, no FIREWORKS_API_KEY, no RUN_SANDBOX_BENCH
gate. Safe to run any time the benchmark JSON already exists.

Usage:
    uv run python scripts/h4_report.py                       # default input
    uv run python scripts/h4_report.py path/to/benchmark.json
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

from bossyk_sandbox.scenarios.runner import FIRING_LABELS
from bossyk_sandbox.scoring.interrupt import H4Result, InterruptRecord, h4_result

REPO_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = REPO_ROOT / "docs" / "bench_output"
DEFAULT_INPUT_PATH = OUTPUT_DIR / "phase2c_orthogonality.json"
OUTPUT_PATH = OUTPUT_DIR / "phase3_h4.json"


def _slow_detected(row: dict[str, object]) -> bool:
    """A slow instrument "fired" (caught the step post-hoc, too late to
    prevent it) if either the drift or policy verdict's label is one of
    scenarios.runner.FIRING_LABELS."""
    drift = row.get("drift")
    policy = row.get("policy")
    drift_fires = isinstance(drift, dict) and drift.get("label") in FIRING_LABELS
    policy_fires = isinstance(policy, dict) and policy.get("label") in FIRING_LABELS
    return drift_fires or policy_fires


def _to_interrupt_records(rows: list[dict[str, object]]) -> list[InterruptRecord]:
    """Rebuilds InterruptRecords from persisted per-step rows. Rows with a
    null outcome_violation (no A1 ground-truth key for that step) are
    skipped, mirroring to_membership/to_gate_outcomes in scenarios/runner.py
    and scripts/benchmark_run.py's live-run equivalent."""
    records: list[InterruptRecord] = []
    for row in rows:
        is_violation = row.get("outcome_violation")
        if is_violation is None:
            continue
        assert isinstance(is_violation, bool)
        records.append(
            InterruptRecord(
                is_violation=is_violation,
                gate_blocked=row.get("gate_verdict") == "block",
                slow_detected=_slow_detected(row),
            )
        )
    return records


def _group_rows_by_domain(rows: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        domain = row["domain"]
        assert isinstance(domain, str)
        grouped.setdefault(domain, []).append(row)
    return grouped


def _h4_report(result: H4Result) -> dict[str, object]:
    low, high = result.prevention_ci95()
    return {
        **asdict(result),
        "prevention_rate": result.prevention_rate(),
        "prevention_ci_low": low,
        "prevention_ci_high": high,
    }


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


def main() -> None:
    input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_INPUT_PATH
    data = json.loads(input_path.read_text())
    rows: list[dict[str, object]] = data["steps"]

    grouped_rows = _group_rows_by_domain(rows)
    domain_names = sorted(grouped_rows)

    all_records = _to_interrupt_records(rows)
    records_by_domain = {
        domain_name: _to_interrupt_records(domain_rows)
        for domain_name, domain_rows in grouped_rows.items()
    }

    print(f"=== H4 interrupt efficacy (from {input_path}) ===")
    combined_result = h4_result(all_records)
    _print_h4_read("Combined", combined_result)
    for domain_name in domain_names:
        _print_h4_read(domain_name, h4_result(records_by_domain[domain_name]))

    report: dict[str, object] = {
        "source": str(input_path),
        "combined": _h4_report(combined_result),
        "per_domain": {
            domain_name: _h4_report(h4_result(records_by_domain[domain_name]))
            for domain_name in domain_names
        },
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(report, indent=2))
    print(f"\nWrote H4 report to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
