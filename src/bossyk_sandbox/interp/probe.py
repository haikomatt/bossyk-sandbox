"""Linear probe over activations -> policy-violation label (numpy, held-out).

The Phase-4 capstone's off-pod half: given the (X, y) matrix `activation_capture`
assembles for one (layer, timepoint) cell, fit a linear probe and ask whether a
policy violation is linearly decodable from the residual stream -- the
policy-compliance analogue of the auditk drift probe.

Two things this module refuses to let you fool yourself with:

- **Held-out evaluation.** d_model is thousands, samples are few, so *training*
  AUROC is ~1.0 by memorisation and meaningless. Every reported number is on a
  held-out split (`probe_auroc`), scored with the same rank AUROC the behavioral
  layer uses (`interp.correlate.auroc`).
- **Controls (from the drift-probe experiment).** `shuffled_label_auroc` must sit
  near 0.5 or the pipeline is leaking; `probe_report` scores several label
  vectors on the SAME activations so the **general-failure confound** is explicit
  -- if a "step is a general error" label probes as well as the policy label, the
  probe is detecting incoherence, not policy violation, and the result is void.

numpy only (already a dep); the fit is a small regularised logistic regression by
gradient descent -- enough for a linear probe, and it keeps the default test
suite free of sklearn/torch. torch/nnsight belong to the pod-only capture script.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from bossyk_sandbox.interp.correlate import auroc

Array = NDArray[np.float64]
IntArray = NDArray[np.intp]


@dataclass(frozen=True)
class LinearProbe:
    """A fitted linear probe: weights + bias, plus the train-set standardisation
    (mean/std) that must be reapplied to any inputs scored later."""

    weights: Array
    bias: float
    mean: Array
    std: Array


def _sigmoid(z: Array) -> Array:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def train_probe(
    x: list[list[float]] | Array,
    y: list[bool] | Array,
    *,
    l2: float = 1.0,
    iters: int = 800,
    lr: float = 0.5,
) -> LinearProbe:
    """Fit a regularised logistic-regression probe. Standardises features on the
    training data (stored for scoring), then gradient-descends the L2-penalised
    logistic loss (penalty on weights, not bias). Deterministic."""
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    if xa.ndim != 2:
        raise ValueError("x must be 2-D (n_samples, n_features)")
    n, d = xa.shape
    mean = xa.mean(axis=0)
    std = xa.std(axis=0)
    std[std == 0.0] = 1.0  # constant features carry no signal; avoid /0
    xs = (xa - mean) / std

    w = np.zeros(d, dtype=np.float64)
    b = 0.0
    for _ in range(iters):
        p = _sigmoid(xs @ w + b)
        err = p - ya
        grad_w = xs.T @ err / n + l2 * w / n
        grad_b = float(err.mean())
        w -= lr * grad_w
        b -= lr * grad_b
    return LinearProbe(weights=w, bias=b, mean=mean, std=std)


def probe_scores(probe: LinearProbe, x: list[list[float]] | Array) -> Array:
    """Violation probability per row, applying the probe's train-set standardisation."""
    xa = np.asarray(x, dtype=np.float64)
    xs = (xa - probe.mean) / probe.std
    return _sigmoid(xs @ probe.weights + probe.bias)


def _split(n: int, *, test_frac: float, seed: int) -> tuple[IntArray, IntArray]:
    idx = np.random.default_rng(seed).permutation(n)
    n_test = max(1, int(round(n * test_frac)))
    return idx[n_test:], idx[:n_test]  # train, test


def probe_auroc(
    x: list[list[float]] | Array,
    y: list[bool] | Array,
    *,
    test_frac: float = 0.3,
    seed: int = 0,
    l2: float = 1.0,
) -> float:
    """Held-out AUROC of a linear probe: deterministic split, fit on train, score
    test, rank-AUROC on the test labels. Returns nan when either the train or the
    test split is single-class (AUROC undefined) -- honest, not a spurious 0.5."""
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=bool)
    if len(xa) < 2:
        return float("nan")
    train_idx, test_idx = _split(len(xa), test_frac=test_frac, seed=seed)
    if len(set(ya[train_idx].tolist())) < 2 or len(set(ya[test_idx].tolist())) < 2:
        return float("nan")
    probe = train_probe(xa[train_idx], ya[train_idx], l2=l2)
    scores = probe_scores(probe, xa[test_idx])
    return auroc(scores.tolist(), ya[test_idx].tolist())


def shuffled_label_auroc(
    x: list[list[float]] | Array,
    y: list[bool] | Array,
    *,
    seed: int = 0,
    l2: float = 1.0,
) -> float:
    """Control: probe held-out AUROC with the labels shuffled. Must sit near 0.5;
    a high value means the evaluation is leaking (e.g. train/test contamination)
    and every other number here is suspect."""
    ya = np.asarray(y, dtype=bool)
    shuffled = np.random.default_rng(seed).permutation(ya)
    return probe_auroc(x, shuffled, seed=seed, l2=l2)


def probe_report(
    x: list[list[float]] | Array,
    labels_by_name: dict[str, list[bool]],
    *,
    seed: int = 0,
    l2: float = 1.0,
) -> dict[str, float]:
    """Held-out probe AUROC for several label vectors over the SAME activations.

    The general-failure confound made explicit: pass e.g.
    {"policy": <policy labels>, "error": <is-general-error labels>,
     "shuffled": <shuffled policy labels>}. The policy probe is only meaningful
    if it beats both the shuffled floor AND the error probe -- otherwise the
    residual encodes 'this step is bad', not 'this step violates policy'."""
    return {
        name: probe_auroc(x, labels, seed=seed, l2=l2) for name, labels in labels_by_name.items()
    }


def _pca_fit(x_train: Array, k: int) -> tuple[Array, Array]:
    """Fit PCA on training rows only (no leakage): center, SVD, keep the top-k
    right-singular vectors as components. Returns (mean, components (k, d)). k is
    clamped to the number of components SVD can produce."""
    mean = x_train.mean(axis=0)
    _u, _s, vt = np.linalg.svd(x_train - mean, full_matrices=False)
    k = min(k, vt.shape[0])
    return mean, vt[:k]


def _pca_apply(x: Array, mean: Array, components: Array) -> Array:
    """Project rows onto fitted PCA components: (x - train_mean) @ Vᵀ."""
    return (x - mean) @ components.T


def _stratified_folds(y: NDArray[np.bool_], n_splits: int, seed: int) -> list[IntArray]:
    """Deterministic stratified k-fold: shuffle each class, deal round-robin into
    folds, so every fold holds ~the same positive/negative ratio -- the fix for
    the tiny-N single-class test folds that made a single 30% split unstable."""
    rng = np.random.default_rng(seed)
    folds: list[list[int]] = [[] for _ in range(n_splits)]
    for cls in (np.where(y)[0], np.where(~y)[0]):
        for i, idx in enumerate(rng.permutation(cls)):
            folds[i % n_splits].append(int(idx))
    return [np.array(sorted(f), dtype=np.intp) for f in folds]


def crossval_oof_scores(
    x: list[list[float]] | Array,
    y: list[bool] | Array,
    *,
    n_splits: int = 5,
    l2: float = 1.0,
    n_components: int | None = None,
    seed: int = 0,
) -> tuple[Array, NDArray[np.bool_]]:
    """Stratified k-fold out-of-fold probe scores: each sample is scored exactly
    once, by a probe trained on the OTHER folds. Optional per-fold PCA
    (`n_components`) fit on the training rows only, then the standardised L2
    logistic probe. Returns (oof_scores, y) in original order.

    This is the small-n stability fix: instead of one 30%-holdout AUROC (which
    swung 1.00->0.31 across layers at d=3584>>n=68 because a single split is
    high-variance), every sample gets a held-out prediction, and AUROC over all n
    of them (`probe_auroc_cv`) is far less split-dependent. `n_splits` is clamped
    down to the smaller class count; if that is < 2 the scores are all-nan (a CV
    is undefined)."""
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=bool)
    n = len(xa)
    n_splits = min(n_splits, int(ya.sum()), int((~ya).sum()))
    oof = np.full(n, np.nan, dtype=np.float64)
    if n_splits < 2:
        return oof, ya
    for test_idx in _stratified_folds(ya, n_splits, seed):
        train_mask = np.ones(n, dtype=bool)
        train_mask[test_idx] = False
        x_train, x_test = xa[train_mask], xa[test_idx]
        if n_components is not None:
            mean, comps = _pca_fit(x_train, n_components)
            x_train, x_test = _pca_apply(x_train, mean, comps), _pca_apply(x_test, mean, comps)
        probe = train_probe(x_train, ya[train_mask], l2=l2)
        oof[test_idx] = probe_scores(probe, x_test)
    return oof, ya


def probe_auroc_cv(
    x: list[list[float]] | Array,
    y: list[bool] | Array,
    *,
    n_splits: int = 5,
    l2: float = 1.0,
    n_components: int | None = None,
    seed: int = 0,
) -> float:
    """Cross-validated held-out AUROC: rank-AUROC over the out-of-fold scores.
    `nan` when the CV is undefined (a class too small to stratify)."""
    oof, ya = crossval_oof_scores(
        x, y, n_splits=n_splits, l2=l2, n_components=n_components, seed=seed
    )
    mask = ~np.isnan(oof)
    if mask.sum() == 0:
        return float("nan")
    return auroc(oof[mask].tolist(), ya[mask].tolist())


def bootstrap_auroc_ci(
    scores: Array,
    y: NDArray[np.bool_],
    *,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the AUROC of `scores` vs `y`: resample the
    (score, label) pairs with replacement `n_boot` times, take the central
    `1-alpha` percentile band of the resampled AUROCs. Single-class resamples are
    skipped. Returns (nan, nan) if fewer than 2 valid resamples -- an honest 'no
    interval', not a fake point. This is the confidence the single-split number
    never had."""
    mask = ~np.isnan(scores)
    s, yv = scores[mask], y[mask]
    n = len(s)
    if n < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    boots: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        val = auroc(s[idx].tolist(), yv[idx].tolist())
        if not math.isnan(val):
            boots.append(val)
    if len(boots) < 2:
        return (float("nan"), float("nan"))
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return (lo, hi)


def probe_cv_report(
    x: list[list[float]] | Array,
    labels_by_name: dict[str, list[bool]],
    *,
    n_splits: int = 5,
    l2: float = 1.0,
    n_components: int | None = None,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    """Cross-validated probe report with confidence intervals for several label
    vectors over the SAME activations -- the robust replacement for `probe_report`
    (single split, no CI). Per label: CV AUROC + a bootstrap CI + the positive
    count. The policy label is only a result if its CI clears BOTH the shuffled
    floor AND the coherence-error confound (compare the intervals, not the point
    estimates)."""
    out: dict[str, dict[str, float]] = {}
    for name, labels in labels_by_name.items():
        oof, ya = crossval_oof_scores(
            x, labels, n_splits=n_splits, l2=l2, n_components=n_components, seed=seed
        )
        lo, hi = bootstrap_auroc_ci(oof, ya, n_boot=n_boot, alpha=alpha, seed=seed)
        mask = ~np.isnan(oof)
        point = auroc(oof[mask].tolist(), ya[mask].tolist()) if mask.sum() else float("nan")
        out[name] = {
            "auroc": point,
            "ci_lo": lo,
            "ci_hi": hi,
            "n_pos": float(int(np.asarray(labels, dtype=bool).sum())),
        }
    return out


def logit_lens(
    residual: list[float] | Array,
    unembedding: list[list[float]] | Array,
    *,
    k: int = 5,
) -> list[tuple[int, float]]:
    """Project a residual-stream vector to vocab logits via the unembedding
    (`residual @ W_U`, W_U shape (d_model, vocab)) and return the top-k
    (token_id, logit), highest first -- 'what token is this activation leaning
    toward at this layer'. Pure; the real W_U comes from the model on the pod."""
    r = np.asarray(residual, dtype=np.float64)
    w_u = np.asarray(unembedding, dtype=np.float64)
    logits = r @ w_u
    top = np.argsort(logits)[::-1][:k]
    return [(int(i), float(logits[i])) for i in top]
