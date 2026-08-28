"""Vectorised rank-AUROC, numerically identical to
`bossyk_sandbox.interp.correlate.auroc` (same average-rank Mann-Whitney tie
convention) but implemented with numpy array ops instead of a per-element
Python loop.

Why this exists: step 5 (`scripts/detector_verdict.py`) evaluates the
hierarchical bootstrap's 10,000-resample paired-gap CI (Amendment 2a), which
calls an AUROC function on the order of 10^5 times over evaluation sets of a
few thousand items each. `interp.correlate.auroc`'s per-element Python loop
for tie-rank assignment is the right choice for its own call sites (small
n, called rarely) but is too slow at this call volume. This module is a
drop-in-equivalent used ONLY inside the bootstrap loop; every POINT estimate
(the numbers actually reported) is computed with the canonical
`interp.correlate.auroc` so the reported statistics are provably the same
function the rest of the repo uses. `test_detector_fast_auroc.py` asserts
the two implementations agree to float64 precision across randomised inputs,
including heavy ties.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


def _average_rank(x: Array) -> Array:
    """1-based average rank per element, ties resolved to the mean rank of
    their tie block -- fully vectorised (argsort + `np.bincount`, no Python
    loop over elements or tie groups)."""
    n = len(x)
    order = np.argsort(x, kind="mergesort")
    sorted_x = x[order]

    is_new_group = np.empty(n, dtype=bool)
    is_new_group[0] = True
    if n > 1:
        is_new_group[1:] = sorted_x[1:] != sorted_x[:-1]
    group_id = np.cumsum(is_new_group) - 1

    plain_ranks_sorted = np.arange(1, n + 1, dtype=np.float64)
    group_counts = np.bincount(group_id)
    group_rank_sums = np.bincount(group_id, weights=plain_ranks_sorted)
    group_avg_rank = group_rank_sums / group_counts

    avg_rank_sorted = group_avg_rank[group_id]
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = avg_rank_sorted
    return ranks


def fast_auroc(scores: Array, y: BoolArray) -> float:
    """Rank AUROC = P(positive score > negative score) + 0.5 * P(tie), i.e.
    the Mann-Whitney U statistic normalised by `n_pos * n_neg`. `nan` when
    either class is empty (undefined), matching `interp.correlate.auroc`.
    """
    n = len(scores)
    if n != len(y):
        raise ValueError("scores and y must be the same length")
    n_pos = int(y.sum())
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _average_rank(scores)
    sum_ranks_pos = float(ranks[y].sum())
    u = sum_ranks_pos - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg)
