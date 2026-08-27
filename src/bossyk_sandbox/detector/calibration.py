"""Expected Calibration Error (ECE) for the frozen baseline tables.

Per the pre-registration (`a-fine-tuned-violation-detector-transfers-cross-
domain.md`), baselines are reported with **raw probabilities, no temperature
scaling** -- temperature scaling is reserved for the trained detectors'
post-hoc calibration step (build-order step 5), fit on a train-domain
validation split. Scoring the baselines' raw output here keeps that
comparison honest: the frozen table is what the detector eval must beat both
on discrimination (AUROC) and calibration (ECE) before scaling.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def expected_calibration_error(scores: Array, y: NDArray[np.bool_], *, n_bins: int = 10) -> float:
    """Equal-width-bin ECE: `sum_b (n_b / n) * |acc(b) - conf(b)|` over `n_bins`
    equal-width bins of `[0, 1]`, where `acc(b)` is the empirical violation
    rate in bin `b` and `conf(b)` is the mean predicted score in that bin.
    `scores` are treated as calibrated-if-correct P(violation).

    NaN scores (e.g. an undefined probe fold) are dropped before binning.
    Returns `nan` if no valid scores remain -- an honest 'no ECE', not a
    spurious 0.0."""
    mask = ~np.isnan(scores)
    s, yv = scores[mask], y[mask]
    n = len(s)
    if n == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # right=True + clip on the interior edges puts a score of exactly 0.0 in
    # bin 0 and a score of exactly 1.0 in the last bin (inclusive both ends).
    bin_idx = np.clip(np.digitize(s, edges[1:-1], right=True), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        in_bin = bin_idx == b
        n_b = int(in_bin.sum())
        if n_b == 0:
            continue
        acc = float(yv[in_bin].mean())
        conf = float(s[in_bin].mean())
        ece += (n_b / n) * abs(acc - conf)
    return float(ece)
