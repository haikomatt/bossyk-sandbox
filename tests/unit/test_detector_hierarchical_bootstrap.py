"""Unit tests for the Amendment-2a hierarchical bootstrap machinery."""

from __future__ import annotations

import numpy as np

from bossyk_sandbox.detector.hierarchical_bootstrap import (
    BootstrapResult,
    CellArrays,
    DomainGroups,
    bootstrap_composite_gap,
    cell_gap,
    composite_gap,
    flat_resample_indices,
    hierarchical_resample_indices,
)


def test_hierarchical_resample_equal_size_scenarios_preserves_total_count() -> None:
    groups = {"s0": [0, 1, 2, 3], "s1": [4, 5, 6, 7], "s2": [8, 9, 10, 11]}
    rng = np.random.default_rng(0)
    idx = hierarchical_resample_indices(groups, rng)
    assert len(idx) == 12  # 3 scenarios drawn, each contributing its own (equal) size


def test_hierarchical_resample_indices_stay_within_domain_range() -> None:
    groups = {"s0": [0, 1, 2], "s1": [3, 4, 5, 6, 7], "s2": [8, 9]}
    rng = np.random.default_rng(1)
    for _ in range(20):
        idx = hierarchical_resample_indices(groups, rng)
        assert idx.min() >= 0
        assert idx.max() <= 9


def test_hierarchical_resample_is_deterministic_given_shared_rng_state() -> None:
    groups = {"s0": [0, 1, 2, 3, 4]}
    rng_a = np.random.default_rng(7)
    rng_b = np.random.default_rng(7)
    a = hierarchical_resample_indices(groups, rng_a)
    b = hierarchical_resample_indices(groups, rng_b)
    assert np.array_equal(a, b)


def test_flat_resample_indices_length_and_range() -> None:
    rng = np.random.default_rng(0)
    idx = flat_resample_indices(50, rng)
    assert len(idx) == 50
    assert idx.min() >= 0
    assert idx.max() < 50


def _perfect_vs_random_cell(n: int, seed: int, *, train: str, eval_: str) -> CellArrays:
    rng = np.random.default_rng(seed)
    y = np.array([True] * (n // 2) + [False] * (n - n // 2))
    rng.shuffle(y)
    # Detector: perfectly separated by construction (+ tiny noise so ties don't dominate).
    detector = np.where(y, 1.0, 0.0) + rng.normal(0, 1e-6, n)
    bow = rng.normal(0, 1, n)  # BoW: pure noise, ~chance
    return CellArrays(
        train_domain=train,
        eval_domain=eval_,
        seed_scores={0: detector, 1: detector, 2: detector},
        bow_scores=bow,
        labels=y,
    )


def test_cell_gap_point_estimate_near_expected_for_perfect_vs_random() -> None:
    cell = _perfect_vs_random_cell(400, seed=0, train="a", eval_="b")
    idx = np.arange(400)
    gap = cell_gap(cell, idx)
    assert gap > 0.4  # detector ~1.0, bow ~0.5 -> gap ~0.5


def test_composite_gap_averages_across_cells() -> None:
    cell1 = _perfect_vs_random_cell(200, seed=1, train="a", eval_="c")
    cell2 = _perfect_vs_random_cell(200, seed=2, train="b", eval_="c")
    idx_by_domain = {"c": np.arange(200)}
    composite = composite_gap([cell1, cell2], idx_by_domain)
    g1 = cell_gap(cell1, np.arange(200))
    g2 = cell_gap(cell2, np.arange(200))
    assert abs(composite - (g1 + g2) / 2.0) < 1e-12


def test_bootstrap_composite_gap_deterministic_given_seed() -> None:
    cell = _perfect_vs_random_cell(150, seed=3, train="a", eval_="d")
    groups = {"d": DomainGroups(domain="d", n=150, scenario_indices={"s0": list(range(150))})}
    r1 = bootstrap_composite_gap([cell], groups, n_resamples=200, seed=99, hierarchical=True)
    r2 = bootstrap_composite_gap([cell], groups, n_resamples=200, seed=99, hierarchical=True)
    assert r1 == r2


def test_bootstrap_composite_gap_strong_separation_excludes_zero() -> None:
    cell = _perfect_vs_random_cell(300, seed=4, train="a", eval_="e")
    scenario_indices = {f"s{i}": list(range(i * 30, (i + 1) * 30)) for i in range(10)}
    groups = {"e": DomainGroups(domain="e", n=300, scenario_indices=scenario_indices)}
    result: BootstrapResult = bootstrap_composite_gap(
        [cell], groups, n_resamples=1000, seed=1, hierarchical=True
    )
    assert result.point_gap > 0.4
    assert result.lower > 0.0
    assert result.ci_excludes_zero is True


def test_hierarchical_ci_wider_than_flat_when_scores_cluster_by_scenario() -> None:
    """The whole point of Amendment 2a: when items cluster by scenario (a
    per-scenario random offset shared by all its items), a flat bootstrap
    that ignores clustering understates variance relative to a hierarchical
    one that resamples at the scenario level too."""
    rng = np.random.default_rng(5)
    n_scenarios, per_scenario = 10, 30
    n = n_scenarios * per_scenario
    y = np.tile([True] * (per_scenario // 2) + [False] * (per_scenario // 2), n_scenarios)
    detector = np.empty(n)
    bow = np.empty(n)
    scenario_indices: dict[str, list[int]] = {}
    for s in range(n_scenarios):
        lo, hi = s * per_scenario, (s + 1) * per_scenario
        scenario_indices[f"s{s}"] = list(range(lo, hi))
        # A scenario-level random offset dominates the item-level signal, so
        # resampling scenarios (not just items) is what actually captures
        # the real source of variance.
        offset = rng.normal(0, 3.0)
        detector[lo:hi] = np.where(y[lo:hi], 1.0, 0.0) + offset + rng.normal(0, 0.05, per_scenario)
        bow[lo:hi] = rng.normal(0, 1, per_scenario)
    cell = CellArrays(
        train_domain="a",
        eval_domain="f",
        seed_scores={0: detector, 1: detector, 2: detector},
        bow_scores=bow,
        labels=y,
    )
    groups = {"f": DomainGroups(domain="f", n=n, scenario_indices=scenario_indices)}
    hier = bootstrap_composite_gap([cell], groups, n_resamples=1500, seed=11, hierarchical=True)
    flat = bootstrap_composite_gap([cell], groups, n_resamples=1500, seed=11, hierarchical=False)
    hier_width = hier.upper - hier.lower
    flat_width = flat.upper - flat.lower
    assert hier_width > flat_width


def test_domain_groups_rejects_mismatched_index_count() -> None:
    try:
        DomainGroups(domain="x", n=10, scenario_indices={"s0": [0, 1, 2]})
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def test_cell_arrays_rejects_mismatched_lengths() -> None:
    y = np.array([True, False, True])
    try:
        CellArrays(
            train_domain="a",
            eval_domain="b",
            seed_scores={0: np.array([0.1, 0.2, 0.3])},
            bow_scores=np.array([0.1, 0.2]),  # wrong length
            labels=y,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")
