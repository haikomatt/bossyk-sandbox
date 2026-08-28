"""Unit tests for the mechanical verdict-condition evaluation
(`bossyk_sandbox.detector.verdict_logic`) -- pure numbers in, booleans out,
no data loading."""

from __future__ import annotations

from bossyk_sandbox.detector.verdict_logic import (
    VERDICT_INSUFFICIENT_DATA,
    VERDICT_REFUTED,
    VERDICT_SUPPORTED,
    VERDICT_UNDETERMINED,
    FamilyGapEvidence,
    evaluate_family,
    evaluate_overall,
)


def _evidence(
    family: str,
    *,
    point_gap: float = 0.10,
    mean4_lower: float = 0.02,
    into_lower: float = 0.02,
    out_lower: float = 0.02,
    ece_detector: float = 0.05,
    ece_bow: float = 0.20,
) -> FamilyGapEvidence:
    return FamilyGapEvidence(
        family=family,
        point_gap_mean4=point_gap,
        hier_ci_mean4_lower=mean4_lower,
        hier_ci_into_advice_lower=into_lower,
        hier_ci_out_of_advice_lower=out_lower,
        ece_detector_mean4_post_temperature=ece_detector,
        ece_bow_fixed_mean4=ece_bow,
    )


def test_family_fully_passing_is_supported() -> None:
    fv = evaluate_family(_evidence("qwen_lora"), base_rate_gate_pass=True)
    assert fv.gap_point_ge_0_05 is True
    assert fv.hier_ci_mean4_gt_0 is True
    assert fv.hier_ci_into_advice_gt_0 is True
    assert fv.hier_ci_out_of_advice_gt_0 is True
    assert fv.auroc_gap_condition is True
    assert fv.ece_condition is True
    assert fv.family_supported is True


def test_family_fails_on_point_gap_below_threshold() -> None:
    fv = evaluate_family(_evidence("deberta", point_gap=0.04), base_rate_gate_pass=True)
    assert fv.gap_point_ge_0_05 is False
    assert fv.auroc_gap_condition is False
    assert fv.family_supported is False


def test_family_fails_when_one_direction_ci_includes_zero_amendment1() -> None:
    # Mean-of-4 CI excludes zero but the out-of-advice direction does not --
    # Amendment 1 requires per-direction, so this must not pass.
    fv = evaluate_family(_evidence("deberta", out_lower=-0.01), base_rate_gate_pass=True)
    assert fv.hier_ci_mean4_gt_0 is True
    assert fv.hier_ci_out_of_advice_gt_0 is False
    assert fv.auroc_gap_condition is False
    assert fv.family_supported is False


def test_family_passes_gap_but_fails_ece_is_not_supported() -> None:
    fv = evaluate_family(
        _evidence("deberta", ece_detector=0.30, ece_bow=0.20), base_rate_gate_pass=True
    )
    assert fv.auroc_gap_condition is True
    assert fv.ece_condition is False
    assert fv.family_supported is False


def test_family_base_rate_gate_failure_blocks_support_even_if_auroc_passes() -> None:
    fv = evaluate_family(_evidence("deberta"), base_rate_gate_pass=False)
    assert fv.auroc_gap_condition is True
    assert fv.ece_condition is True
    assert fv.base_rate_gate_pass is False
    assert fv.family_supported is False


def test_overall_supported_when_at_least_one_family_supported() -> None:
    fv_a = evaluate_family(_evidence("deberta"), base_rate_gate_pass=True)
    fv_b = evaluate_family(_evidence("qwen_lora", point_gap=0.01), base_rate_gate_pass=True)
    verdict = evaluate_overall([fv_a, fv_b], base_rate_gate_pass_overall=True)
    assert verdict == VERDICT_SUPPORTED


def test_overall_refuted_when_both_families_fail_auroc_gap_condition() -> None:
    fv_a = evaluate_family(_evidence("deberta", point_gap=0.01), base_rate_gate_pass=True)
    fv_b = evaluate_family(_evidence("qwen_lora", mean4_lower=-0.02), base_rate_gate_pass=True)
    verdict = evaluate_overall([fv_a, fv_b], base_rate_gate_pass_overall=True)
    assert verdict == VERDICT_REFUTED


def test_overall_insufficient_data_caps_regardless_of_auroc_numbers() -> None:
    fv_a = evaluate_family(_evidence("deberta"), base_rate_gate_pass=True)  # would be SUPPORTED
    fv_b = evaluate_family(_evidence("qwen_lora"), base_rate_gate_pass=True)
    verdict = evaluate_overall([fv_a, fv_b], base_rate_gate_pass_overall=False)
    assert verdict == VERDICT_INSUFFICIENT_DATA


def test_overall_undetermined_when_gap_passes_but_ece_fails_for_all_passing_families() -> None:
    # deberta passes the AUROC-gap condition but fails ECE; qwen_lora fails
    # the AUROC-gap condition outright. Neither SUPPORTED (ECE required) nor
    # REFUTED (deberta's auroc_gap_condition is True, so "both families fail
    # the gap condition" is false) applies.
    fv_a = evaluate_family(
        _evidence("deberta", ece_detector=0.30, ece_bow=0.20), base_rate_gate_pass=True
    )
    fv_b = evaluate_family(_evidence("qwen_lora", point_gap=0.01), base_rate_gate_pass=True)
    verdict = evaluate_overall([fv_a, fv_b], base_rate_gate_pass_overall=True)
    assert verdict == VERDICT_UNDETERMINED
