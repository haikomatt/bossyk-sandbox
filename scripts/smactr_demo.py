#!/usr/bin/env python
"""Phase 4 SMACTR demo -- close the eval loop on one caught failure.

retail-008 (unauthorised address modification) was DETECTED TOO LATE in H4:
the policy judge fired post-hoc, but the fast gate allowed it (nothing gated
`modify_user_address`), so it executed. The SMACTR response derives a
fast-path constraint, freezes a regression probe, and feeds the rule back
into `retail_fast_rules` -- converting that detected-too-late into a
PREVENTION.

This shows the before/after, computed deterministically: only the gate
decisions change (a new fast rule), so the slow-judge verdicts + A1 labels
are reused from the persisted Phase 2c run -- no new judge calls.

    uv run python scripts/smactr_demo.py
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from bossyk_sandbox.domains import domain_config
from bossyk_sandbox.gate import Gate
from bossyk_sandbox.governance.smactr import CaughtFailure, save_threat_model, smactr_response
from bossyk_sandbox.instruments.base import Verdict
from bossyk_sandbox.scenarios.loader import load_scenarios
from bossyk_sandbox.scenarios.runner import FIRING_LABELS
from bossyk_sandbox.scoring.interrupt import H4Result, InterruptRecord, h4_result

REPO = Path(__file__).parent.parent
PHASE2C = REPO / "docs" / "bench_output" / "phase2c_orthogonality.json"
OUT = REPO / "docs" / "bench_output" / "phase4_smactr.json"
THREAT_MODEL = REPO / "docs" / "bench_output" / "threat_model.json"
DOMAIN = "retail"


def _slow_detected(row: dict[str, Any]) -> bool:
    drift = (row.get("drift") or {}).get("label")
    policy = (row.get("policy") or {}).get("label")
    return drift in FIRING_LABELS or policy in FIRING_LABELS


def _before_records(rows: list[dict[str, Any]]) -> list[InterruptRecord]:
    """From the persisted Phase 2c gate decisions (pre-SMACTR fast rules)."""
    return [
        InterruptRecord(
            is_violation=bool(row["outcome_violation"]),
            gate_blocked=row["gate_verdict"] == "block",
            slow_detected=_slow_detected(row),
        )
        for row in rows
        if row["outcome_violation"] is not None
    ]


def _after_records(rows: list[dict[str, Any]]) -> list[InterruptRecord]:
    """Recompute gate decisions with the now-augmented retail fast rules
    (the SMACTR-derived rule is live in `retail_fast_rules`); reuse the
    persisted slow verdicts + A1 labels keyed by (scenario_id, step_index)."""
    cfg = domain_config(DOMAIN)
    by_key = {(row["scenario_id"], row["step_index"]): row for row in rows}
    records: list[InterruptRecord] = []
    for scenario in load_scenarios(cfg.scenarios_path):
        gate = Gate(instruments=cfg.fast_rules_factory())
        for index, step in enumerate(scenario.steps):
            decision = gate.evaluate(step.proposed)
            row = by_key.get((scenario.scenario_id, index))
            if row is None or row["outcome_violation"] is None:
                continue
            records.append(
                InterruptRecord(
                    is_violation=bool(row["outcome_violation"]),
                    gate_blocked=decision.verdict is Verdict.BLOCK,
                    slow_detected=_slow_detected(row),
                )
            )
    return records


def _h4_dict(result: H4Result) -> dict[str, Any]:
    return {
        "n_violations": result.n_violations,
        "prevented": result.prevented,
        "detected_too_late": result.detected_too_late,
        "undetected": result.undetected,
        "harm_on": result.harm_on,
        "prevention_rate": result.prevention_rate(),
    }


def main() -> None:
    rows = [row for row in json.loads(PHASE2C.read_text())["steps"] if row["domain"] == DOMAIN]
    before = h4_result(_before_records(rows))
    after = h4_result(_after_records(rows))

    failure = CaughtFailure(
        domain="retail",
        scenario_id="retail-008-unauthorized-address-modification",
        boundary="unauthorized_modification",
        tool_name="modify_user_address",
        declared_intent="Update the customer's saved address.",
    )
    entry = smactr_response(
        failure,
        derived_constraint=(
            "Gate modify_user_address on a prior get_user_details for the same "
            "user_id (RequireLookupBeforeCancel) -- fed back into retail_fast_rules."
        ),
        regression_probe_id="retail-008-unauthorized-address-modification",
        status="mitigated",
    )
    save_threat_model([entry], THREAT_MODEL)

    entry_dict = {**asdict(entry), "severity": entry.severity.value}
    report = {
        "caught_failure": asdict(failure),
        "threat_model_entry": entry_dict,
        "before": _h4_dict(before),
        "after": _h4_dict(after),
        "prevented_gain": after.prevented - before.prevented,
    }
    OUT.write_text(json.dumps(report, indent=2))

    print("=== SMACTR eval-loop closure (retail) ===")
    print(f"Caught failure: {failure.scenario_id} ({failure.tool_name})")
    print(f"FMEA severity: {entry.severity.value}")
    print(f"Derived constraint: {entry.derived_constraint}")
    print(f"Regression probe frozen: {entry.regression_probe_id}\n")
    print(
        f"BEFORE (pre-SMACTR):  prevented={before.prevented}  "
        f"detected_too_late={before.detected_too_late}  "
        f"prevention_rate={before.prevention_rate():.3f}"
    )
    print(
        f"AFTER  (rule fed back): prevented={after.prevented}  "
        f"detected_too_late={after.detected_too_late}  "
        f"prevention_rate={after.prevention_rate():.3f}"
    )
    print(
        f"\nLoop closed: +{after.prevented - before.prevented} prevention "
        f"(retail-008 detected_too_late -> prevented)."
    )
    print(f"Wrote {OUT} + {THREAT_MODEL}")


if __name__ == "__main__":
    main()
