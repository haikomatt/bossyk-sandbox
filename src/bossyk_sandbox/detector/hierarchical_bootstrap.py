"""Hierarchical bootstrap for the paired (detector - BoW) AUROC gap
(Amendment 2a, adopted 2026-08-27): "scenario (with replacement, within
domain) -> decision (with replacement, within scenario), 10,000 resamples,
seeds recorded." The flat bootstrap (treats every decision as independent)
is computed alongside for transparency; the hierarchical CI is the
DECISION-BEARING one per the amendment.

A "cell" here is one (train_domain, eval_domain) pair scored by a detector
family: a paired detector-minus-baseline AUROC gap, where "detector" is the
mean over 3 seeds' AUROC (this module does not itself decide how seeds are
combined -- `cell_gap` takes an already-assembled per-seed score dict and
means the per-seed AUROCs; see `scripts/detector_verdict.py`'s module
docstring for why seed-averaging happens per-replicate rather than seeds
being a third resampling level -- Amendment 2a's hierarchy is explicitly
two levels, scenario then decision, with no seed level named).

A "composite" is a mean over several cells that share resampling where they
share an eval domain (e.g. the into-advice direction = mean of
{retail->advice, airline->advice}; both cells score the SAME
advice-eligibility items, so a single resample of advice-eligibility's
scenario/decision structure is drawn once per bootstrap replicate and reused
for both cells -- not resampled twice independently, which would understate
the correlation between the two cells' noise).

Uses `fast_auroc` (numpy-vectorised, proven equal to
`interp.correlate.auroc` to float64 precision in `test_detector_fast_auroc.py`)
for the ~10^5 AUROC evaluations a 10,000-resample composite bootstrap needs;
every REPORTED point estimate is still computed with the canonical
`interp.correlate.auroc` elsewhere and cross-checked against this module's
`point_gap` (see `scripts/detector_verdict.py`).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from bossyk_sandbox.detector.fast_auroc import fast_auroc

Array = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.intp]


@dataclass(frozen=True)
class DomainGroups:
    """One eval domain's item count and its scenario -> item-index grouping,
    over the SAME canonical item order every `CellArrays` sharing this
    `domain` name must use."""

    domain: str
    n: int
    scenario_indices: dict[str, list[int]]

    def __post_init__(self) -> None:
        total = sum(len(v) for v in self.scenario_indices.values())
        if total != self.n:
            raise ValueError(
                f"domain {self.domain!r}: scenario_indices cover {total} items, expected n={self.n}"
            )


@dataclass(frozen=True)
class CellArrays:
    """Aligned per-item arrays for one (train_domain, eval_domain) cell.
    `seed_scores`, `bow_scores`, and `labels` all share one row order --
    the SAME order as the `DomainGroups` for `eval_domain`."""

    train_domain: str
    eval_domain: str
    seed_scores: dict[int, Array]
    bow_scores: Array
    labels: BoolArray

    def __post_init__(self) -> None:
        n = len(self.labels)
        if len(self.bow_scores) != n:
            raise ValueError("bow_scores length must match labels length")
        for seed, scores in self.seed_scores.items():
            if len(scores) != n:
                raise ValueError(f"seed {seed} scores length must match labels length")


@dataclass(frozen=True)
class BootstrapResult:
    point_gap: float
    lower: float
    upper: float
    n_resamples: int
    seed: int
    hierarchical: bool
    ci_excludes_zero: bool
    replicate_mean: float


def hierarchical_resample_indices(
    scenario_indices: dict[str, list[int]], rng: np.random.Generator
) -> IntArray:
    """One hierarchical resample: draw `len(scenario_indices)` scenarios
    with replacement, then within each drawn scenario draw its own item
    count with replacement. Vectorised per scenario (one `rng.integers` call
    per drawn scenario, not per item)."""
    keys = list(scenario_indices.keys())
    n_scenarios = len(keys)
    if n_scenarios == 0:
        return np.array([], dtype=np.intp)
    chosen = rng.integers(0, n_scenarios, size=n_scenarios)
    parts: list[IntArray] = []
    for c in chosen:
        pool = np.asarray(scenario_indices[keys[c]], dtype=np.intp)
        m = len(pool)
        draw = rng.integers(0, m, size=m)
        parts.append(pool[draw])
    return np.concatenate(parts)


def flat_resample_indices(n: int, rng: np.random.Generator) -> IntArray:
    """One flat (pseudoreplicated) resample: draw `n` items with
    replacement, ignoring scenario structure -- reported alongside the
    hierarchical CI for transparency (Amendment 2a), never decision-bearing."""
    return rng.integers(0, n, size=n)


def cell_gap(cell: CellArrays, idx: IntArray) -> float:
    """Paired (mean-over-seeds detector AUROC) - (BoW AUROC) on the item
    subset `idx` (with repeats, for a bootstrap resample; `np.arange(n)` for
    the point estimate)."""
    y = cell.labels[idx]
    seed_aurocs = [fast_auroc(scores[idx], y) for scores in cell.seed_scores.values()]
    mean_seed_auroc = float(np.mean(seed_aurocs))
    bow_auroc = fast_auroc(cell.bow_scores[idx], y)
    return mean_seed_auroc - bow_auroc


def composite_gap(cells: Sequence[CellArrays], idx_by_domain: dict[str, IntArray]) -> float:
    """Mean cell_gap over `cells`, each scored against its own eval domain's
    resampled index set (cells sharing an eval domain share that domain's
    draw)."""
    gaps = [cell_gap(c, idx_by_domain[c.eval_domain]) for c in cells]
    return float(np.mean(gaps))


def bootstrap_composite_gap(
    cells: Sequence[CellArrays],
    domain_groups: dict[str, DomainGroups],
    *,
    n_resamples: int,
    seed: int,
    hierarchical: bool,
) -> BootstrapResult:
    """95% percentile bootstrap CI on the composite paired gap over `cells`.
    One resample per UNIQUE eval domain among `cells` per replicate, shared
    by every cell scoring that domain within that replicate."""
    if not cells:
        raise ValueError("cells must be non-empty")
    unique_domains = sorted({c.eval_domain for c in cells})
    for d in unique_domains:
        if d not in domain_groups:
            raise ValueError(f"missing DomainGroups for eval_domain={d!r}")

    point_idx = {d: np.arange(domain_groups[d].n, dtype=np.intp) for d in unique_domains}
    point = composite_gap(cells, point_idx)

    rng = np.random.default_rng(seed)
    replicate_values = np.empty(n_resamples, dtype=np.float64)
    for b in range(n_resamples):
        idx_by_domain: dict[str, IntArray] = {}
        for d in unique_domains:
            g = domain_groups[d]
            if hierarchical:
                idx_by_domain[d] = hierarchical_resample_indices(g.scenario_indices, rng)
            else:
                idx_by_domain[d] = flat_resample_indices(g.n, rng)
        replicate_values[b] = composite_gap(cells, idx_by_domain)

    lower, upper = (float(v) for v in np.percentile(replicate_values, [2.5, 97.5]))
    return BootstrapResult(
        point_gap=point,
        lower=lower,
        upper=upper,
        n_resamples=n_resamples,
        seed=seed,
        hierarchical=hierarchical,
        ci_excludes_zero=lower > 0.0,
        replicate_mean=float(np.mean(replicate_values)),
    )
