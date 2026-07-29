"""Hermetic tests for interp.probe -- numpy only, synthetic activations."""

from __future__ import annotations

import math

import numpy as np

from bossyk_sandbox.interp.probe import (
    logit_lens,
    probe_auroc,
    probe_report,
    shuffled_label_auroc,
)


def _separable(n: int = 100, d: int = 8, *, seed: int = 0) -> tuple[list[list[float]], list[bool]]:
    """Half violation / half compliant, cleanly separated along dim 0."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 0.5, size=(n, d))
    y = [i < n // 2 for i in range(n)]
    x[: n // 2, 0] += 3.0  # violations high on dim 0
    x[n // 2 :, 0] -= 3.0  # compliant low
    return x.tolist(), y


def test_probe_recovers_a_linearly_separable_signal_on_heldout() -> None:
    x, y = _separable()
    assert probe_auroc(x, y) > 0.9  # held-out, so this is real separation not memorisation


def test_shuffled_labels_sit_near_chance() -> None:
    x, y = _separable()
    v = shuffled_label_auroc(x, y)
    assert 0.25 <= v <= 0.75, f"shuffled control should be ~0.5, got {v}"


def test_probe_auroc_single_class_is_nan() -> None:
    x, _ = _separable()
    assert math.isnan(probe_auroc(x, [True] * len(x)))


def test_probe_auroc_too_few_samples_is_nan() -> None:
    assert math.isnan(probe_auroc([[0.0, 1.0]], [True]))


def test_probe_report_separates_policy_from_shuffled() -> None:
    x, y = _separable()
    shuffled = np.random.default_rng(1).permutation(np.asarray(y, dtype=bool)).tolist()
    report = probe_report(x, {"policy": y, "shuffled": shuffled})
    assert report["policy"] > 0.9
    assert report["shuffled"] <= 0.75
    assert set(report) == {"policy", "shuffled"}


def test_probe_report_confound_a_second_signal_is_visible() -> None:
    # A "general-failure" label separable on a DIFFERENT dim should also probe
    # high -- the whole point of reporting it beside the policy label so a real
    # run can check the policy signal isn't just detecting that.
    x, y = _separable()
    xa = np.asarray(x)
    error = xa[:, 1] > xa[:, 1].mean()  # separable on dim 1
    report = probe_report(x, {"policy": y, "error": error.tolist()})
    assert report["policy"] > 0.9
    assert not math.isnan(report["error"])


def test_logit_lens_returns_topk_highest_first() -> None:
    residual = [1.0, 2.0, 3.0]
    unembedding = np.eye(3).tolist()  # logits == residual
    top = logit_lens(residual, unembedding, k=2)
    assert top[0] == (2, 3.0)
    assert top[1] == (1, 2.0)
