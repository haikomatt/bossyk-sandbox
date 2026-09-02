"""Mechanical evaluation of the pre-registered verdict conditions
(`Hypotheses/a-fine-tuned-violation-detector-transfers-cross-domain.md`,
Amendments 1 and 2). Pure functions over already-computed numbers -- no I/O,
no data loading; `scripts/detector_verdict.py` supplies the evidence.

**Every condition below is transcribed, not invented.** Two aggregation
choices the pre-registration text does not pin down exactly are named
explicitly as flagged conventions (not silently resolved) -- see
`ECE_AGGREGATION_NOTE` and `UNDETERMINED`'s docstring. Both are reported as
booleans in the output JSON alongside the raw numbers, so a reviewer can
recompute either interpretation.

Original threshold ("supported"): on the advice-crossing cells, for at least
one detector family --
  (a) point gap `mean AUROC(detector) - mean AUROC(BoW) >= 0.05`, AND
  (b) the bootstrap 95% CI on the paired gap has lower bound `> 0`, AND
  (c) `ECE(detector, post-temperature) < ECE(BoW)` on the same cells.

Amendment 1 (2026-08-26) tightens (b): the CI lower-bound-`>0` requirement
now applies to the 4-cell mean AND to each of the two directions
(into-advice, out-of-advice) separately -- "a supported verdict then
requires... the paired-gap CI lower bound `> 0` in EACH direction, not only
on the mean." Strictly tightens the original: anything passing the new
condition passes the old.

Amendment 2b(ii)+(iii) (2026-08-27): the >=150-positives-per-OOD-cell power
floor is a HARD GATE, not a target. An underpowered cell is not scored;
if any advice-crossing cell's OOD evaluation set (the tested domain's FULL
unique corpus) is underpowered, the verdict is capped at INSUFFICIENT-DATA.

Refutes if: the paired-gap CI includes 0 or the point gap `< 0.05` on the
advice-crossing cells, for BOTH families. Note this refuting condition is
stated in the dossier purely in terms of the AUROC gap/CI -- it does not
mention ECE.
"""

from __future__ import annotations

from dataclasses import dataclass

GAP_THRESHOLD = 0.05

VERDICT_SUPPORTED = "SUPPORTED"
VERDICT_REFUTED = "REFUTED"
VERDICT_INSUFFICIENT_DATA = "INSUFFICIENT-DATA"
# Not a pre-registered category. The dossier names exactly two exit
# conditions (SUPPORTED's three-part AND, REFUTED's AUROC-gap-only OR). It
# does not say what to call a family that passes the AUROC-gap condition
# (a, b) but fails the ECE condition (c) -- that family is neither
# SUPPORTED (ECE required) nor does it trigger REFUTED (REFUTED's own text
# is silent on ECE, so an ECE failure alone cannot refute). UNDETERMINED
# names this gap in the pre-registration explicitly rather than silently
# forcing the case into SUPPORTED or REFUTED. Flagged for Matt's ruling.
VERDICT_UNDETERMINED = "UNDETERMINED"

# The ECE condition's text -- "ECE(detector, post-temperature) < ECE(BoW)
# on the same cells" -- does not say whether "on the cells" means a per-cell
# check (all 4 advice-crossing cells individually) or an aggregate (mean
# ECE over the 4 cells, mirroring how the AUROC condition aggregates the
# same 4 cells into one mean). This module implements the AGGREGATE
# (4-cell-mean) reading, for consistency with how the AUROC point-gap
# condition itself aggregates the same 4 cells -- flagged, not silently
# assumed authoritative; `scripts/detector_verdict.py` also reports each
# cell's ECE individually so the per-cell reading can be checked by hand.
ECE_AGGREGATION_NOTE = (
    "ECE condition evaluated as a 4-cell MEAN (detector mean ECE < BoW-fixed mean ECE "
    "over the 4 advice-crossing cells), mirroring the AUROC point-gap condition's own "
    "aggregation. The pre-registration text does not disambiguate mean-vs-per-cell; "
    "per-cell ECE values are reported alongside so the per-cell reading is also checkable."
)


@dataclass(frozen=True)
class FamilyGapEvidence:
    """The numbers `evaluate_family` needs for one detector family, already
    computed by `scripts/detector_verdict.py` from the bootstrap + ECE
    tables."""

    family: str
    point_gap_mean4: float
    hier_ci_mean4_lower: float
    hier_ci_into_advice_lower: float
    hier_ci_out_of_advice_lower: float
    ece_detector_mean4_post_temperature: float
    ece_bow_fixed_mean4: float


@dataclass(frozen=True)
class FamilyVerdict:
    family: str
    gap_point_ge_0_05: bool
    hier_ci_mean4_gt_0: bool
    hier_ci_into_advice_gt_0: bool
    hier_ci_out_of_advice_gt_0: bool
    auroc_gap_condition: bool
    ece_condition: bool
    base_rate_gate_pass: bool
    family_supported: bool


def evaluate_family(
    evidence: FamilyGapEvidence,
    *,
    base_rate_gate_pass: bool,
    gap_threshold: float = GAP_THRESHOLD,
) -> FamilyVerdict:
    """Every named boolean is one transcribed clause of the pre-registration
    (see module docstring); `auroc_gap_condition` is their conjunction per
    Amendment 1, and `family_supported` is the full three-part AND plus the
    Amendment-2b(ii) hard gate."""
    gap_point_ge_0_05 = evidence.point_gap_mean4 >= gap_threshold
    hier_ci_mean4_gt_0 = evidence.hier_ci_mean4_lower > 0.0
    hier_ci_into_advice_gt_0 = evidence.hier_ci_into_advice_lower > 0.0
    hier_ci_out_of_advice_gt_0 = evidence.hier_ci_out_of_advice_lower > 0.0
    auroc_gap_condition = (
        gap_point_ge_0_05
        and hier_ci_mean4_gt_0
        and hier_ci_into_advice_gt_0
        and hier_ci_out_of_advice_gt_0
    )
    ece_condition = evidence.ece_detector_mean4_post_temperature < evidence.ece_bow_fixed_mean4
    family_supported = auroc_gap_condition and ece_condition and base_rate_gate_pass
    return FamilyVerdict(
        family=evidence.family,
        gap_point_ge_0_05=gap_point_ge_0_05,
        hier_ci_mean4_gt_0=hier_ci_mean4_gt_0,
        hier_ci_into_advice_gt_0=hier_ci_into_advice_gt_0,
        hier_ci_out_of_advice_gt_0=hier_ci_out_of_advice_gt_0,
        auroc_gap_condition=auroc_gap_condition,
        ece_condition=ece_condition,
        base_rate_gate_pass=base_rate_gate_pass,
        family_supported=family_supported,
    )


def evaluate_overall(
    family_verdicts: list[FamilyVerdict], *, base_rate_gate_pass_overall: bool
) -> str:
    """Combine per-family verdicts into the single mechanical top-level
    category. Order matters: the Amendment-2b(iii) data-sufficiency gate is
    checked FIRST and, if failed, caps the verdict regardless of what the
    AUROC numbers say -- "if any advice-crossing cell is underpowered, the
    verdict is capped at 'insufficient data'"."""
    if not base_rate_gate_pass_overall:
        return VERDICT_INSUFFICIENT_DATA
    if any(fv.family_supported for fv in family_verdicts):
        return VERDICT_SUPPORTED
    if all(not fv.auroc_gap_condition for fv in family_verdicts):
        return VERDICT_REFUTED
    return VERDICT_UNDETERMINED
