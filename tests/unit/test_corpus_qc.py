"""Hermetic tests for `bossyk_sandbox.interp.corpus_qc`
(phase-detector-training-step2-datagen.md Part A, item 3: cross-split/
cross-domain leakage scan, class-balance report, hand-audit sample export,
provenance/honesty manifest). Pure/offline -- no network, no LLM.
"""

from __future__ import annotations

from bossyk_sandbox.interp.corpus_qc import (
    build_manifest,
    class_balance_report,
    cross_domain_near_duplicate_scan,
    export_hand_audit_sample,
    leakage_scan,
)


def _row(
    scenario_id: str,
    prompt: str,
    *,
    domain: str = "retail",
    is_violation: bool = False,
    persona_id: str | None = None,
) -> dict[str, object]:
    return {
        "row_id": f"{domain}:{scenario_id}#turn-0",
        "task_id": f"{domain}:{scenario_id}:0",
        "domain": domain,
        "scenario_id": scenario_id,
        "persona_id": persona_id,
        "variant_index": 0,
        "step_id": "turn-0",
        "prompt": prompt,
        "is_violation": is_violation,
        "action": None,
        "generator": "stub",
        "empty": False,
    }


# --- leakage_scan ----------------------------------------------------------


def test_leakage_scan_clean_when_groups_dont_cross_splits() -> None:
    splits = {
        "train": [_row("s1", "a"), _row("s2", "b")],
        "val": [_row("s3", "c")],
        "test": [_row("s4", "d")],
    }
    report = leakage_scan(splits, group_fields=("scenario_id",))
    assert report.clean
    assert report.leaked_groups == []


def test_leakage_scan_catches_a_scenario_id_that_crosses_splits() -> None:
    # Deliberately broken input (assembly should never produce this; QC
    # independently re-verifies rather than trusting the splitter).
    splits = {
        "train": [_row("s1", "a")],
        "val": [_row("s1", "a-variant")],  # same scenario_id, different split
        "test": [],
    }
    report = leakage_scan(splits, group_fields=("scenario_id",))
    assert not report.clean
    assert len(report.leaked_groups) == 1
    assert report.leaked_groups[0].key == ("s1",)
    assert set(report.leaked_groups[0].splits) == {"train", "val"}


def test_leakage_scan_checks_persona_id_too_when_requested() -> None:
    splits = {
        "train": [_row("s1", "a", persona_id="ADV-0001")],
        "val": [_row("s2", "b", persona_id="ADV-0001")],  # same persona, diff scenario
        "test": [],
    }
    report = leakage_scan(splits, group_fields=("scenario_id", "persona_id"))
    assert not report.clean
    assert any(g.key == ("ADV-0001",) for g in report.leaked_groups)


# --- cross_domain_near_duplicate_scan --------------------------------------


def test_cross_domain_near_duplicate_scan_flags_near_identical_prompts_across_domains() -> None:
    records = [
        _row("r1", "cancel my pending order right now please", domain="retail"),
        _row("a1", "cancel my pending order right now please ok", domain="airline"),
        _row("r2", "what is my order status", domain="retail"),
    ]
    report = cross_domain_near_duplicate_scan(records, threshold=0.6, shingle_size=3)
    assert report.n_pairs_checked > 0
    assert report.rate > 0.0
    assert any(p.domain_a != p.domain_b for p in report.pairs)


def test_cross_domain_near_duplicate_scan_zero_when_domains_are_distinct() -> None:
    records = [
        _row("r1", "cancel my pending retail order right now", domain="retail"),
        _row(
            "a1",
            "please explain the eligibility criteria for this benefit scheme",
            domain="advice-eligibility",
        ),
    ]
    report = cross_domain_near_duplicate_scan(records, threshold=0.9)
    assert report.rate == 0.0
    assert report.pairs == []


# --- class_balance_report ---------------------------------------------------


def test_class_balance_report_counts_per_domain_and_split() -> None:
    splits = {
        "train": [
            _row("s1", "a", domain="retail", is_violation=True),
            _row("s2", "b", domain="retail"),
        ],
        "val": [_row("s3", "c", domain="airline", is_violation=True)],
        "test": [_row("s4", "d", domain="airline")],
    }
    report = class_balance_report(splits)
    assert report["retail"]["train"]["n"] == 2
    assert report["retail"]["train"]["violations"] == 1
    assert report["retail"]["train"]["violation_rate"] == 0.5
    assert report["airline"]["val"]["n"] == 1
    assert report["airline"]["test"]["violations"] == 0


# --- export_hand_audit_sample -----------------------------------------------


def test_export_hand_audit_sample_is_deterministic_and_bounded() -> None:
    records = [_row(f"s{i}", f"prompt {i}") for i in range(200)]
    sample_a = export_hand_audit_sample(records, n=50, seed=0)
    sample_b = export_hand_audit_sample(records, n=50, seed=0)
    assert len(sample_a) == 50
    assert [r["scenario_id"] for r in sample_a] == [r["scenario_id"] for r in sample_b]


def test_export_hand_audit_sample_caps_at_available_records() -> None:
    records = [_row(f"s{i}", f"prompt {i}") for i in range(10)]
    sample = export_hand_audit_sample(records, n=50, seed=0)
    assert len(sample) == 10


# --- build_manifest ----------------------------------------------------------


def test_build_manifest_carries_provenance_and_honesty_fields() -> None:
    manifest = build_manifest(
        domain="retail",
        generator_model="stub-test",
        serving_path="stub",
        seed=0,
        counts={"train": 8, "val": 1, "test": 1},
        class_balance={"train": 0.25, "val": 0.0, "test": 1.0},
        cap_hit=None,
        near_dup_rate_within_domain=0.0,
        generated_at="2026-08-26T00:00:00+00:00",
    )
    assert manifest["domain"] == "retail"
    assert manifest["generator_model"] == "stub-test"
    assert manifest["cap_hit"] is None
    assert manifest["cap_was_hit"] is False
    assert manifest["counts"]["train"] == 8


def test_build_manifest_reports_cap_hit_honestly() -> None:
    manifest = build_manifest(
        domain="retail",
        generator_model="stub-test",
        serving_path="stub",
        seed=0,
        counts={"train": 8, "val": 1, "test": 1},
        class_balance={"train": 0.25, "val": 0.0, "test": 1.0},
        cap_hit="usd",
        near_dup_rate_within_domain=0.0,
        generated_at="2026-08-26T00:00:00+00:00",
    )
    assert manifest["cap_hit"] == "usd"
    assert manifest["cap_was_hit"] is True
