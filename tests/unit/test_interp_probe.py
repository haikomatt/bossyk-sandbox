"""Hermetic tests for interp.probe -- numpy only, synthetic activations."""

from __future__ import annotations

import math

import numpy as np

from bossyk_sandbox.interp.probe import (
    bootstrap_auroc_ci,
    crossval_oof_scores,
    logit_lens,
    probe_auroc,
    probe_auroc_cv,
    probe_cv_report,
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


def _highdim_separable(
    n: int = 120, d: int = 200, *, seed: int = 0
) -> tuple[list[list[float]], list[bool]]:
    """Half violation / half compliant, separable on dim 0, with many noise dims
    (d >> per-class n) -- the d_model>>n regime that broke the single-split probe."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, size=(n, d))
    y = [i < n // 2 for i in range(n)]
    x[: n // 2, 0] += 2.5
    x[n // 2 :, 0] -= 2.5
    return x.tolist(), y


def test_cv_recovers_a_separable_signal() -> None:
    x, y = _separable()
    assert probe_auroc_cv(x, y, n_splits=5) > 0.9


def test_cv_shuffled_labels_sit_near_chance() -> None:
    x, y = _separable()
    shuffled = np.random.default_rng(3).permutation(np.asarray(y, dtype=bool)).tolist()
    assert probe_auroc_cv(x, shuffled) < 0.75


def test_cv_undefined_when_class_too_small_is_nan() -> None:
    # one positive cannot be stratified across folds -> CV undefined
    x, _ = _separable(n=20)
    y = [True] + [False] * 19
    assert math.isnan(probe_auroc_cv(x, y, n_splits=5))


def test_cv_with_pca_recovers_signal_in_high_dim() -> None:
    # d=200 >> per-class n=60: PCA to 30 comps still recovers the dim-0 signal
    x, y = _highdim_separable()
    assert probe_auroc_cv(x, y, n_splits=5, n_components=30) > 0.85


def test_oof_scores_cover_every_sample_once() -> None:
    x, y = _separable(n=100)
    oof, ya = crossval_oof_scores(x, y, n_splits=5)
    assert len(oof) == 100
    assert not np.isnan(oof).any()  # every sample got a held-out score
    assert ya.tolist() == y


def test_bootstrap_ci_brackets_a_strong_signal_above_chance() -> None:
    x, y = _separable()
    oof, ya = crossval_oof_scores(x, y, n_splits=5)
    lo, hi = bootstrap_auroc_ci(oof, ya, n_boot=500, seed=0)
    assert 0.5 < lo <= hi <= 1.0  # a real signal's whole interval clears chance


def test_bootstrap_ci_single_sample_is_nan() -> None:
    lo, hi = bootstrap_auroc_ci(np.array([0.5]), np.array([True]))
    assert math.isnan(lo) and math.isnan(hi)


def test_cv_report_policy_ci_clears_the_shuffled_floor() -> None:
    x, y = _separable()
    shuffled = np.random.default_rng(1).permutation(np.asarray(y, dtype=bool)).tolist()
    report = probe_cv_report(x, {"policy": y, "shuffled": shuffled}, n_boot=500)
    assert report["policy"]["ci_lo"] > report["shuffled"]["ci_hi"]  # intervals separate
    assert report["policy"]["n_pos"] == 50.0
    assert set(report["policy"]) == {"auroc", "ci_lo", "ci_hi", "n_pos"}


def test_logit_lens_returns_topk_highest_first() -> None:
    residual = [1.0, 2.0, 3.0]
    unembedding = np.eye(3).tolist()  # logits == residual
    top = logit_lens(residual, unembedding, k=2)
    assert top[0] == (2, 3.0)
    assert top[1] == (1, 2.0)
