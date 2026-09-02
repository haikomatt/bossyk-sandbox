"""Temperature scaling for the trained detectors' post-hoc calibration
(phase-detector-training.md build-order step 5's calibration measurement;
fit location is Amendment 3's calibration slice, not train/val/test).

Pure numpy, no torch/sklearn/scipy: temperature scaling here is a single
scalar `T` fit by minimising the mean binary negative-log-likelihood of
`sigmoid(logit / T)` against the true labels, via a bounded golden-section
search (deterministic, no gradient framework needed for a 1-parameter fit).

Baselines (`bow_baseline.py`) are deliberately NOT touched by this module --
per the dossier, baselines report raw probabilities only; temperature
scaling is reserved for the trained detectors.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]

# Search bounds for T. T=1 is "no rescaling"; values this far from 1 already
# represent extreme over/under-confidence, so the search bracket is
# generous without letting a pathological fit run away to +-inf.
T_MIN = 0.05
T_MAX = 20.0
_GOLDEN_TOL = 1e-4
_EPS = 1e-12


def apply_temperature(logits: Array, temperature: float) -> Array:
    """`sigmoid(logits / temperature)`, i.e. the calibrated P(violation)."""
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    return 1.0 / (1.0 + np.exp(-logits / temperature))


def negative_log_likelihood(logits: Array, y: NDArray[np.bool_], temperature: float) -> float:
    """Mean binary NLL of `apply_temperature(logits, temperature)` against
    `y`. Probabilities are clipped away from 0/1 so a perfectly-confident
    wrong prediction gives a large but finite loss, not `inf`/`nan`."""
    if len(logits) == 0:
        return float("nan")
    p = np.clip(apply_temperature(logits, temperature), _EPS, 1.0 - _EPS)
    yf = y.astype(np.float64)
    nll = -(yf * np.log(p) + (1.0 - yf) * np.log(1.0 - p))
    return float(nll.mean())


def fit_temperature(
    logits: Array,
    y: NDArray[np.bool_],
    *,
    t_min: float = T_MIN,
    t_max: float = T_MAX,
    tol: float = _GOLDEN_TOL,
) -> float:
    """Fit the scalar temperature minimising mean NLL via golden-section
    search over `[t_min, t_max]`. Deterministic (no randomness, no seed
    needed) -- the same `(logits, y)` always yields the same `T`.

    Degenerate inputs (empty slice, or a slice with fewer than 2 items) fall
    back to `T=1.0` (no rescaling) rather than fitting noise -- callers
    (Amendment 3's calibration slice can be tiny) should treat that as "not
    enough signal to calibrate", not as a real fit."""
    if len(logits) < 2:
        return 1.0

    invphi = (np.sqrt(5.0) - 1.0) / 2.0  # 1/phi
    invphi2 = (3.0 - np.sqrt(5.0)) / 2.0  # 1/phi^2
    a, b = t_min, t_max
    h = b - a
    if h <= tol:
        return (a + b) / 2.0

    n_steps = int(np.ceil(np.log(tol / h) / np.log(invphi)))
    c = a + invphi2 * h
    d = a + invphi * h
    fc = negative_log_likelihood(logits, y, c)
    fd = negative_log_likelihood(logits, y, d)

    for _ in range(max(n_steps, 1)):
        if fc < fd:
            b, d, fd = d, c, fc
            h = invphi * h
            c = a + invphi2 * h
            fc = negative_log_likelihood(logits, y, c)
        else:
            a, c, fc = c, d, fd
            h = invphi * h
            d = a + invphi * h
            fd = negative_log_likelihood(logits, y, d)
        if h <= tol:
            break

    best_t, best_nll = c, fc
    if fd < best_nll:
        best_t, best_nll = d, fd
    return float(best_t)
