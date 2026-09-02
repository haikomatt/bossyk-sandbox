#!/usr/bin/env python
"""Pre-registered evaluation (phase-detector-training.md build-order step 5):
mechanically evaluates every locked condition in
`Hypotheses/a-fine-tuned-violation-detector-transfers-cross-domain.md`
(Amendments 1-3) against the frozen baselines and the 18-cell / 54-file
trained-detector scores. Hermetic: no API calls, no pods, no spend -- pure
numpy over already-committed artefacts.

MANDATORY built-in check: regenerated per-cell BoW AUROC (both vectorisers,
all 9 cells) must match `probes/detector/results/baselines_frozen.json` to
within 1e-9, or this script aborts (`SystemExit`) before computing anything
else -- the baseline table is the ground the paired gap is measured against,
so silent drift there would invalidate every downstream number.

Outputs:
- `probes/detector/results/verdict.json` -- every number and every named
  condition boolean, fully auditable.
- `docs/phase-detector-verdict.md` -- matrices, gaps, CIs, ECE, base rates,
  and the mechanical verdict statement ONLY. No editorialising, no words
  like "unfortunately" / "impressively" / "as expected" -- numbers only.

Two conventions this script fixes because the pre-registration text does
not pin them down exactly. Both are FLAGGED here and in the output, not
silently decided:

1. **Seed treatment inside the bootstrap.** The dossier lists seeds and the
   bootstrap as two separate mechanisms ("3 seeds per config; bootstrap 95%
   CI on the paired... gap"), and Amendment 2a's hierarchical spec is
   explicitly two levels (scenario -> decision) -- no seed level is named.
   This script reads that as: at every bootstrap replicate, the detector's
   AUROC on the resampled item set is the MEAN of its 3 seeds' AUROC on
   that same resample -- the direct generalisation of "per family, mean
   over 3 seeds" (the point-estimate convention) into each replicate, with
   only the EVALUATION SET resampled, never the seed. Per-seed AUROC is
   reported separately (not folded into the resampling) so this choice is
   independently checkable.
2. **ECE condition aggregation** -- see `verdict_logic.ECE_AGGREGATION_NOTE`.

Both conventions, and every verdict condition, are transcribed as named
booleans/notes in `verdict.json` so a reviewer can recompute either reading
by hand from the reported per-cell numbers.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from bossyk_sandbox.detector.base_rate import wilson_interval
from bossyk_sandbox.detector.fast_auroc import fast_auroc
from bossyk_sandbox.detector.hierarchical_bootstrap import (
    CellArrays,
    DomainGroups,
    flat_resample_indices,
    hierarchical_resample_indices,
)
from bossyk_sandbox.detector.transfer_eval import (
    BowRegen,
    DetectorCellStats,
    EvalDomainAssembly,
    assemble_eval_domain,
    build_cell_arrays,
    detector_point_stats,
    load_detector_scores,
    load_domains,
    regenerate_bow_scores,
    verify_bow_regen_against_frozen,
)
from bossyk_sandbox.detector.verdict_logic import (
    FamilyGapEvidence,
    evaluate_family,
    evaluate_overall,
)

REPO_ROOT = Path(__file__).parent.parent
DATA_ROOT = REPO_ROOT / "probes" / "detector" / "data"
RESULTS_ROOT = REPO_ROOT / "probes" / "detector" / "results"
SCORES_ROOT = RESULTS_ROOT / "scores"
FROZEN_PATH = RESULTS_ROOT / "baselines_frozen.json"
VERDICT_JSON_PATH = RESULTS_ROOT / "verdict.json"
VERDICT_DOC_PATH = REPO_ROOT / "docs" / "phase-detector-verdict.md"

DOMAINS: tuple[str, ...] = ("retail", "airline", "advice-eligibility")
FAMILIES: tuple[str, ...] = ("deberta", "qwen_lora")
SEEDS: tuple[int, ...] = (0, 1, 2)
POWER_GATE = 150
N_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260828  # recorded per Amendment 2a ("seeds recorded")

# name -> (train_domain, eval_domain) cells contributing to it.
CELL_NAMES: tuple[str, ...] = (
    "retail->advice",
    "airline->advice",
    "advice->retail",
    "advice->airline",
    "retail->airline",
    "airline->retail",
)
CELL_OF: dict[str, tuple[str, str]] = {
    "retail->advice": ("retail", "advice-eligibility"),
    "airline->advice": ("airline", "advice-eligibility"),
    "advice->retail": ("advice-eligibility", "retail"),
    "advice->airline": ("advice-eligibility", "airline"),
    "retail->airline": ("retail", "airline"),
    "airline->retail": ("airline", "retail"),
}
ADVICE_CROSSING_CELLS = ("retail->advice", "airline->advice", "advice->retail", "advice->airline")
INTO_ADVICE_CELLS = ("retail->advice", "airline->advice")
OUT_OF_ADVICE_CELLS = ("advice->retail", "advice->airline")
SATURATION_CELLS = ("retail->airline", "airline->retail")
COMPOSITES: dict[str, tuple[str, ...]] = {
    "mean_advice_crossing_4cell": ADVICE_CROSSING_CELLS,
    "into_advice": INTO_ADVICE_CELLS,
    "out_of_advice": OUT_OF_ADVICE_CELLS,
    "saturation_control_mean": SATURATION_CELLS,
}


def log(msg: str) -> None:
    print(msg, file=sys.stderr)


def cell_point_gap(cell: CellArrays) -> float:
    idx = np.arange(len(cell.labels))
    y = cell.labels[idx]
    seed_aurocs = [fast_auroc(scores[idx], y) for scores in cell.seed_scores.values()]
    bow = fast_auroc(cell.bow_scores[idx], y)
    return float(np.mean(seed_aurocs)) - bow


def bootstrap_all_family_gaps(
    cell_arrays: dict[str, CellArrays],
    domain_groups: dict[str, DomainGroups],
    *,
    n_resamples: int,
    seed: int,
    hierarchical: bool,
) -> dict[str, np.ndarray]:
    """One shared replicate loop producing hierarchical (or flat) bootstrap
    replicate values for every single cell in `CELL_NAMES` AND every
    composite in `COMPOSITES`, for one detector family. All cells and
    composites here only ever touch the 3 domains in `DOMAINS` as eval
    domains, so each replicate draws each domain's resample ONCE and reuses
    it for every cell/composite that scores that domain within the same
    replicate -- avoiding 1) independent domain resamples per composite
    (which would be statistically wasteful) and 2) treating shared-domain
    cells as independent (which would be statistically wrong).

    Returns `{name: replicate_values}` (numpy arrays of length
    `n_resamples`) for every entry in `CELL_NAMES` and `COMPOSITES`.
    """
    unique_domains = sorted({cell_arrays[name].eval_domain for name in CELL_NAMES})
    rng = np.random.default_rng(seed)
    replicate_values: dict[str, list[float]] = {
        name: [] for name in list(CELL_NAMES) + list(COMPOSITES)
    }

    for _ in range(n_resamples):
        idx_by_domain: dict[str, np.ndarray] = {}
        for d in unique_domains:
            g = domain_groups[d]
            idx_by_domain[d] = (
                hierarchical_resample_indices(g.scenario_indices, rng)
                if hierarchical
                else flat_resample_indices(g.n, rng)
            )
        cell_gaps: dict[str, float] = {}
        for name in CELL_NAMES:
            cell = cell_arrays[name]
            idx = idx_by_domain[cell.eval_domain]
            y = cell.labels[idx]
            seed_aurocs = [fast_auroc(scores[idx], y) for scores in cell.seed_scores.values()]
            bow = fast_auroc(cell.bow_scores[idx], y)
            gap = float(np.mean(seed_aurocs)) - bow
            cell_gaps[name] = gap
            replicate_values[name].append(gap)
        for composite_name, members in COMPOSITES.items():
            replicate_values[composite_name].append(float(np.mean([cell_gaps[m] for m in members])))

    return {name: np.array(values, dtype=np.float64) for name, values in replicate_values.items()}


def ci_from_replicates(point: float, replicates: np.ndarray) -> dict[str, float]:
    """95% percentile CI, excluding replicates where the gap was UNDEFINED
    (a resample landed on a single-class subset -- AUROC is undefined
    there; `fast_auroc` returns `nan`, matching `interp.correlate.auroc`'s
    convention). Real on this data: 5 of retail's 10 v2 scenarios have zero
    violations, so any composite scoring retail can hit this by chance.
    `np.nanpercentile`/`np.nanmean` exclude those replicates rather than
    letting a single `nan` silently turn the whole CI into `nan`
    (`np.percentile`'s default propagation) -- `n_undefined_replicates` is
    reported so the exclusion is auditable, not silent."""
    n_undefined = int(np.isnan(replicates).sum())
    lower, upper = (float(v) for v in np.nanpercentile(replicates, [2.5, 97.5]))
    return {
        "point_gap": point,
        "lower": lower,
        "upper": upper,
        "ci_excludes_zero": lower > 0.0,
        "replicate_mean": float(np.nanmean(replicates)),
        "n_undefined_replicates": float(n_undefined),
    }


def build_family_cell_arrays(
    *,
    family: str,
    domains: dict[str, Any],
    assemblies: dict[str, EvalDomainAssembly],
    detector_scores: dict[str, Any],
    regen: BowRegen,
) -> dict[str, CellArrays]:
    result: dict[str, CellArrays] = {}
    for name, (train_d, eval_d) in CELL_OF.items():
        detector_by_seed = {s: detector_scores[family][train_d][s][eval_d] for s in SEEDS}
        result[name] = build_cell_arrays(
            train_domain=train_d,
            eval_domain=eval_d,
            assembly=assemblies[eval_d],
            detector_by_seed=detector_by_seed,
            bow_scores_by_row_id=regen.per_item["fixed_vocab"][train_d][eval_d],
        )
    return result


def cell_matrix_entry(
    *,
    family: str,
    train_d: str,
    eval_d: str,
    domains: dict[str, Any],
    detector_scores: dict[str, Any],
    regen: BowRegen,
) -> dict[str, Any]:
    in_domain = train_d == eval_d
    rows = domains[eval_d].test if in_domain else domains[eval_d].full
    detector_by_seed = {s: detector_scores[family][train_d][s][eval_d] for s in SEEDS}
    stats: DetectorCellStats = detector_point_stats(rows, detector_by_seed)
    bow_fixed = regen.cells["fixed_vocab"][train_d][eval_d]
    bow_hashing = regen.cells["hashing"][train_d][eval_d]
    return {
        "in_domain": in_domain,
        "eval_set": "test_split" if in_domain else "full_unique_corpus",
        "n": stats.n,
        "n_pos": stats.n_pos,
        "underpowered": stats.n_pos < POWER_GATE,
        "detector": {
            "per_seed_auroc": stats.per_seed_auroc,
            "mean_auroc": stats.mean_auroc,
            "per_seed_ece_raw_pre_temperature": stats.per_seed_ece_raw,
            "mean_ece_raw_pre_temperature": stats.mean_ece_raw,
            "per_seed_ece_calibrated_post_temperature": stats.per_seed_ece_calibrated,
            "mean_ece_calibrated_post_temperature": stats.mean_ece_calibrated,
        },
        "bow_fixed_vocab": {"auroc": bow_fixed.auroc, "ece": bow_fixed.ece},
        "bow_hashing": {"auroc": bow_hashing.auroc, "ece": bow_hashing.ece},
        "point_gap_vs_bow_fixed": stats.mean_auroc - bow_fixed.auroc,
        "point_gap_vs_bow_hashing": stats.mean_auroc - bow_hashing.auroc,
    }


def build_full_matrix(
    *, family: str, domains: dict[str, Any], detector_scores: dict[str, Any], regen: BowRegen
) -> dict[str, dict[str, Any]]:
    matrix: dict[str, dict[str, Any]] = {}
    for train_d in DOMAINS:
        matrix[train_d] = {}
        for eval_d in DOMAINS:
            matrix[train_d][eval_d] = cell_matrix_entry(
                family=family,
                train_d=train_d,
                eval_d=eval_d,
                domains=domains,
                detector_scores=detector_scores,
                regen=regen,
            )
    return matrix


def compute_base_rates(domains: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for d in DOMAINS:
        full_n = len(domains[d].full)
        full_pos = sum(1 for r in domains[d].full if r["is_violation"])
        test_n = len(domains[d].test)
        test_pos = sum(1 for r in domains[d].test if r["is_violation"])
        full_ci = wilson_interval(full_pos, full_n)
        test_ci = wilson_interval(test_pos, test_n)
        out[d] = {
            "full_unique_corpus": {
                "n": full_n,
                "n_pos": full_pos,
                "rate": full_ci.rate,
                "ci_lower_95": full_ci.lower,
                "ci_upper_95": full_ci.upper,
                "gate_pass_ge_150_pos": full_pos >= POWER_GATE,
            },
            "test_split": {
                "n": test_n,
                "n_pos": test_pos,
                "rate": test_ci.rate,
                "ci_lower_95": test_ci.lower,
                "ci_upper_95": test_ci.upper,
                "gate_pass_ge_150_pos": test_pos >= POWER_GATE,
            },
        }
    return out


def build_family_report(
    *,
    family: str,
    domains: dict[str, Any],
    assemblies: dict[str, EvalDomainAssembly],
    domain_groups: dict[str, DomainGroups],
    detector_scores: dict[str, Any],
    regen: BowRegen,
    base_rates: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    cell_arrays = build_family_cell_arrays(
        family=family,
        domains=domains,
        assemblies=assemblies,
        detector_scores=detector_scores,
        regen=regen,
    )

    log(f"[{family}] running hierarchical bootstrap ({N_RESAMPLES} resamples)...")
    hier_replicates = bootstrap_all_family_gaps(
        cell_arrays, domain_groups, n_resamples=N_RESAMPLES, seed=BOOTSTRAP_SEED, hierarchical=True
    )
    log(f"[{family}] running flat bootstrap ({N_RESAMPLES} resamples)...")
    flat_replicates = bootstrap_all_family_gaps(
        cell_arrays, domain_groups, n_resamples=N_RESAMPLES, seed=BOOTSTRAP_SEED, hierarchical=False
    )

    points = {name: cell_point_gap(cell_arrays[name]) for name in CELL_NAMES}
    for composite_name, members in COMPOSITES.items():
        points[composite_name] = float(np.mean([points[m] for m in members]))

    gaps: dict[str, Any] = {}
    for name in list(CELL_NAMES) + list(COMPOSITES):
        hier_ci = ci_from_replicates(points[name], hier_replicates[name])
        flat_ci = ci_from_replicates(points[name], flat_replicates[name])
        # The fast (vectorised) point estimate must agree with the
        # canonical-function point estimate to the precision fast_auroc was
        # proven to (test_detector_fast_auroc.py): a self-check that the
        # speed optimisation used inside the bootstrap did not silently
        # diverge from the reported numbers.
        assert abs(hier_ci["point_gap"] - flat_ci["point_gap"]) < 1e-9
        gaps[name] = {
            "point_gap": points[name],
            "hierarchical": {
                "lower_95": hier_ci["lower"],
                "upper_95": hier_ci["upper"],
                "ci_excludes_zero": hier_ci["ci_excludes_zero"],
                "replicate_mean": hier_ci["replicate_mean"],
                "n_undefined_replicates": int(hier_ci["n_undefined_replicates"]),
            },
            "flat": {
                "lower_95": flat_ci["lower"],
                "upper_95": flat_ci["upper"],
                "ci_excludes_zero": flat_ci["ci_excludes_zero"],
                "replicate_mean": flat_ci["replicate_mean"],
                "n_undefined_replicates": int(flat_ci["n_undefined_replicates"]),
            },
        }

    # ECE table on the advice-crossing cells: detector pre/post-T vs BoW (both
    # vectorisers), per-cell and the 4-cell mean (verdict_logic.ECE_AGGREGATION_NOTE).
    ece_rows: dict[str, Any] = {}
    for name in ADVICE_CROSSING_CELLS:
        train_d, eval_d = CELL_OF[name]
        detector_by_seed = {s: detector_scores[family][train_d][s][eval_d] for s in SEEDS}
        stats = detector_point_stats(domains[eval_d].full, detector_by_seed)
        bow_fixed = regen.cells["fixed_vocab"][train_d][eval_d]
        bow_hashing = regen.cells["hashing"][train_d][eval_d]
        ece_rows[name] = {
            "detector_ece_pre_temperature": stats.mean_ece_raw,
            "detector_ece_post_temperature": stats.mean_ece_calibrated,
            "bow_fixed_vocab_ece": bow_fixed.ece,
            "bow_hashing_ece": bow_hashing.ece,
        }
    ece_detector_mean4_post = float(
        np.mean([ece_rows[n]["detector_ece_post_temperature"] for n in ADVICE_CROSSING_CELLS])
    )
    ece_detector_mean4_pre = float(
        np.mean([ece_rows[n]["detector_ece_pre_temperature"] for n in ADVICE_CROSSING_CELLS])
    )
    ece_bow_fixed_mean4 = float(
        np.mean([ece_rows[n]["bow_fixed_vocab_ece"] for n in ADVICE_CROSSING_CELLS])
    )
    ece_bow_hashing_mean4 = float(
        np.mean([ece_rows[n]["bow_hashing_ece"] for n in ADVICE_CROSSING_CELLS])
    )

    base_rate_gate_advice_crossing = all(
        base_rates[eval_d]["full_unique_corpus"]["gate_pass_ge_150_pos"]
        for name in ADVICE_CROSSING_CELLS
        for (_train_d, eval_d) in [CELL_OF[name]]
    )

    evidence = FamilyGapEvidence(
        family=family,
        point_gap_mean4=gaps["mean_advice_crossing_4cell"]["point_gap"],
        hier_ci_mean4_lower=gaps["mean_advice_crossing_4cell"]["hierarchical"]["lower_95"],
        hier_ci_into_advice_lower=gaps["into_advice"]["hierarchical"]["lower_95"],
        hier_ci_out_of_advice_lower=gaps["out_of_advice"]["hierarchical"]["lower_95"],
        ece_detector_mean4_post_temperature=ece_detector_mean4_post,
        ece_bow_fixed_mean4=ece_bow_fixed_mean4,
    )
    family_verdict = evaluate_family(evidence, base_rate_gate_pass=base_rate_gate_advice_crossing)

    return {
        "matrix": build_full_matrix(
            family=family, domains=domains, detector_scores=detector_scores, regen=regen
        ),
        "gaps": gaps,
        "ece": {
            "per_cell": ece_rows,
            "mean4_detector_pre_temperature": ece_detector_mean4_pre,
            "mean4_detector_post_temperature": ece_detector_mean4_post,
            "mean4_bow_fixed_vocab": ece_bow_fixed_mean4,
            "mean4_bow_hashing": ece_bow_hashing_mean4,
        },
        "evidence": {
            "point_gap_mean4": evidence.point_gap_mean4,
            "hier_ci_mean4_lower": evidence.hier_ci_mean4_lower,
            "hier_ci_into_advice_lower": evidence.hier_ci_into_advice_lower,
            "hier_ci_out_of_advice_lower": evidence.hier_ci_out_of_advice_lower,
            "ece_detector_mean4_post_temperature": evidence.ece_detector_mean4_post_temperature,
            "ece_bow_fixed_mean4": evidence.ece_bow_fixed_mean4,
        },
        "conditions": {
            "gap_point_ge_0_05": family_verdict.gap_point_ge_0_05,
            "hier_ci_mean4_gt_0": family_verdict.hier_ci_mean4_gt_0,
            "hier_ci_into_advice_gt_0": family_verdict.hier_ci_into_advice_gt_0,
            "hier_ci_out_of_advice_gt_0": family_verdict.hier_ci_out_of_advice_gt_0,
            "auroc_gap_condition": family_verdict.auroc_gap_condition,
            "ece_condition": family_verdict.ece_condition,
            "base_rate_gate_pass": family_verdict.base_rate_gate_pass,
            "family_supported": family_verdict.family_supported,
        },
        "_family_verdict": family_verdict,
    }


def render_matrix_table(matrix: dict[str, dict[str, Any]], *, key: str, label: str) -> str:
    lines = [f"**{label}**", "", "| train \\ eval | " + " | ".join(DOMAINS) + " |"]
    lines.append("|---|" + "---|" * len(DOMAINS))
    for train_d in DOMAINS:
        cells = []
        for eval_d in DOMAINS:
            entry = matrix[train_d][eval_d]
            if key == "detector_auroc":
                v = entry["detector"]["mean_auroc"]
            elif key == "bow_fixed_auroc":
                v = entry["bow_fixed_vocab"]["auroc"]
            elif key == "bow_hashing_auroc":
                v = entry["bow_hashing"]["auroc"]
            else:
                raise ValueError(key)
            flag = " (u)" if entry["underpowered"] else ""
            cells.append(f"{v:.4f}{flag}" if not np.isnan(v) else "nan")
        lines.append(f"| {train_d} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("(u) = underpowered (n_pos < 150 in that cell's evaluation set)")
    return "\n".join(lines)


def render_doc(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Detector-training pre-registered evaluation (build-order step 5)")
    lines.append("")
    lines.append(f"Generated: {report['generated_at']}")
    lines.append(
        f"Bootstrap: {N_RESAMPLES} resamples, seed={BOOTSTRAP_SEED}, hierarchical "
        "(scenario->decision, Amendment 2a) is decision-bearing; flat reported alongside."
    )
    lines.append("")
    lines.append("## BoW regeneration check")
    lines.append("")
    lines.append(
        f"Cells checked: {report['bow_regen_check']['n_cells_checked']}. "
        f"Mismatches (tolerance 1e-9): {report['bow_regen_check']['n_mismatches']}. "
        f"Result: {'PASSED' if report['bow_regen_check']['passed'] else 'FAILED'}."
    )
    lines.append("")
    lines.append("## Base rates (Amendment 2b(i))")
    lines.append("")
    lines.append("| domain | eval_set | n | n_pos | rate | 95% CI | gate (>=150 pos) |")
    lines.append("|---|---|---|---|---|---|---|")
    for d in DOMAINS:
        for eval_set_key, eval_set_label in (
            ("full_unique_corpus", "full_unique_corpus"),
            ("test_split", "test_split"),
        ):
            br = report["base_rates"][d][eval_set_key]
            gate = br["gate_pass_ge_150_pos"]
            ci = f"[{br['ci_lower_95']:.4f}, {br['ci_upper_95']:.4f}]"
            row = f"| {d} | {eval_set_label} | {br['n']} | {br['n_pos']} | {br['rate']:.4f} | "
            lines.append(row + f"{ci} | {gate} |")
    lines.append("")

    for family in FAMILIES:
        fam = report["families"][family]
        lines.append(f"## Family: {family}")
        lines.append("")
        lines.append("### Full 3x3 transfer matrix")
        lines.append("")
        lines.append(
            render_matrix_table(fam["matrix"], key="detector_auroc", label=f"{family} AUROC")
        )
        lines.append("")
        lines.append(
            render_matrix_table(
                fam["matrix"], key="bow_fixed_auroc", label="BoW (fixed-vocab) AUROC"
            )
        )
        lines.append("")
        lines.append(
            render_matrix_table(fam["matrix"], key="bow_hashing_auroc", label="BoW (hashing) AUROC")
        )
        lines.append("")

        lines.append("### Per-seed detector AUROC (advice-crossing + saturation cells)")
        lines.append("")
        lines.append("| cell | seed0 | seed1 | seed2 | mean |")
        lines.append("|---|---|---|---|---|")
        for name in list(ADVICE_CROSSING_CELLS) + list(SATURATION_CELLS):
            train_d, eval_d = CELL_OF[name]
            entry = fam["matrix"][train_d][eval_d]
            per_seed = entry["detector"]["per_seed_auroc"]
            lines.append(
                f"| {name} | {per_seed[0]:.4f} | {per_seed[1]:.4f} | {per_seed[2]:.4f} | "
                f"{entry['detector']['mean_auroc']:.4f} |"
            )
        lines.append("")

        lines.append("### Paired (detector - BoW fixed-vocab) AUROC gap, with bootstrap 95% CI")
        lines.append("")
        lines.append(
            "| cell/composite | point gap | hier CI | hier excl. 0 | flat CI | flat excl. 0 |"
        )
        lines.append("|---|---|---|---|---|---|")
        for name in list(ADVICE_CROSSING_CELLS) + list(SATURATION_CELLS) + list(COMPOSITES):
            g = fam["gaps"][name]
            h, fl = g["hierarchical"], g["flat"]
            lines.append(
                f"| {name} | {g['point_gap']:.4f} | [{h['lower_95']:.4f}, {h['upper_95']:.4f}] | "
                f"{h['ci_excludes_zero']} | [{fl['lower_95']:.4f}, {fl['upper_95']:.4f}] | "
                f"{fl['ci_excludes_zero']} |"
            )
        lines.append("")

        lines.append("### ECE table (advice-crossing cells)")
        lines.append("")
        lines.append("| cell | detector pre-T | detector post-T | BoW fixed-vocab | BoW hashing |")
        lines.append("|---|---|---|---|---|")
        for name in ADVICE_CROSSING_CELLS:
            e = fam["ece"]["per_cell"][name]
            lines.append(
                f"| {name} | {e['detector_ece_pre_temperature']:.4f} | "
                f"{e['detector_ece_post_temperature']:.4f} | {e['bow_fixed_vocab_ece']:.4f} | "
                f"{e['bow_hashing_ece']:.4f} |"
            )
        e4 = fam["ece"]
        lines.append(
            f"| **mean (4-cell)** | {e4['mean4_detector_pre_temperature']:.4f} | "
            f"{e4['mean4_detector_post_temperature']:.4f} | {e4['mean4_bow_fixed_vocab']:.4f} | "
            f"{e4['mean4_bow_hashing']:.4f} |"
        )
        lines.append("")

        lines.append("### Mechanical condition evaluation")
        lines.append("")
        for k, v in fam["conditions"].items():
            lines.append(f"- `{k}`: {v}")
        lines.append("")

    lines.append("## Overall mechanical verdict")
    lines.append("")
    lines.append(f"`{report['overall_verdict']}`")
    lines.append("")
    lines.append(
        "Category definitions: SUPPORTED / REFUTED are the two pre-registered exit "
        "conditions; INSUFFICIENT-DATA is Amendment 2b(iii)'s hard gate; UNDETERMINED "
        "is not pre-registered -- used only when the AUROC-gap condition passes for at "
        "least one family but the ECE condition fails for every family that passes it "
        "(see `verdict_logic.VERDICT_UNDETERMINED` docstring)."
    )
    lines.append("")
    lines.append("## Flagged implementation conventions (not silently resolved)")
    lines.append("")
    lines.append(
        "1. Bootstrap seed treatment: detector AUROC per replicate = mean of 3 seeds' "
        "AUROC on that replicate's resampled evaluation set; only the evaluation set is "
        "resampled, seeds are not a resampling level (Amendment 2a names two levels only)."
    )
    lines.append(
        "2. ECE condition aggregation: evaluated as a 4-cell MEAN (detector post-T mean "
        "ECE < BoW-fixed mean ECE), mirroring the AUROC point-gap condition's own "
        "aggregation. Per-cell ECE is reported above so the per-cell reading is also "
        "checkable by hand."
    )
    lines.append("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    global N_RESAMPLES, BOOTSTRAP_SEED  # noqa: PLW0603
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--n-resamples", type=int, default=N_RESAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    args = parser.parse_args(argv)
    N_RESAMPLES = args.n_resamples
    BOOTSTRAP_SEED = args.bootstrap_seed

    frozen = json.loads(FROZEN_PATH.read_text())
    log("Loading v2 domain splits...")
    domains = load_domains(DOMAINS, data_root=DATA_ROOT, corpus_version="v2")

    log("Regenerating BoW per-item scores (fixed-vocab + hashing)...")
    regen = regenerate_bow_scores(domains, seed=frozen["bow"]["config"]["seed"])
    mismatches = verify_bow_regen_against_frozen(regen, frozen)
    n_cells_checked = 2 * len(DOMAINS) * len(DOMAINS)
    if mismatches:
        for m in mismatches:
            log(f"MISMATCH: {m}")
        raise SystemExit(
            f"BoW regeneration check FAILED: {len(mismatches)}/{n_cells_checked} cell(s) "
            "mismatched baselines_frozen.json by > 1e-9. Aborting before computing anything "
            "downstream (dossier: no verdict without a verified frozen baseline)."
        )
    log(
        f"BoW regeneration check PASSED: {n_cells_checked}/{n_cells_checked} cells "
        "match frozen baselines (<1e-9)."
    )

    log("Loading committed detector scores (54 files)...")
    detector_scores = load_detector_scores(
        SCORES_ROOT, families=FAMILIES, train_domains=DOMAINS, seeds=SEEDS, eval_domains=DOMAINS
    )

    assemblies = {d: assemble_eval_domain(domains[d]) for d in DOMAINS}
    domain_groups = {d: assemblies[d].groups for d in DOMAINS}
    base_rates = compute_base_rates(domains)

    families_out: dict[str, Any] = {}
    family_verdicts = []
    for family in FAMILIES:
        fam_report = build_family_report(
            family=family,
            domains=domains,
            assemblies=assemblies,
            domain_groups=domain_groups,
            detector_scores=detector_scores,
            regen=regen,
            base_rates=base_rates,
        )
        family_verdicts.append(fam_report.pop("_family_verdict"))
        families_out[family] = fam_report

    base_rate_gate_pass_overall = all(
        base_rates[d]["full_unique_corpus"]["gate_pass_ge_150_pos"] for d in DOMAINS
    )
    overall_verdict = evaluate_overall(
        family_verdicts, base_rate_gate_pass_overall=base_rate_gate_pass_overall
    )

    report: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "corpus_version": "v2",
        "bootstrap": {
            "n_resamples": N_RESAMPLES,
            "seed": BOOTSTRAP_SEED,
            "hierarchy": (
                "scenario (with replacement, within domain) -> "
                "decision (with replacement, within scenario)"
            ),
            "decision_bearing": "hierarchical",
        },
        "bow_regen_check": {
            "passed": len(mismatches) == 0,
            "n_cells_checked": n_cells_checked,
            "n_mismatches": len(mismatches),
            "tolerance": 1e-9,
        },
        "base_rates": base_rates,
        "base_rate_gate_pass_overall": base_rate_gate_pass_overall,
        "families": families_out,
        "overall_verdict": overall_verdict,
    }

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    VERDICT_JSON_PATH.write_text(json.dumps(report, indent=2, allow_nan=True))
    log(f"wrote {VERDICT_JSON_PATH}")

    VERDICT_DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    VERDICT_DOC_PATH.write_text(render_doc(report))
    log(f"wrote {VERDICT_DOC_PATH}")

    log(f"OVERALL VERDICT: {overall_verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
