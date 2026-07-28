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
