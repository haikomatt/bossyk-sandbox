"""Hermetic tests for bossyk_sandbox.detector.transfer_eval -- tiny synthetic
v2 corpora + fake detector score files on disk, no network, no real
`probes/detector/data`."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from bossyk_sandbox.detector.corpus_io import load_all_domains
from bossyk_sandbox.detector.hierarchical_bootstrap import bootstrap_composite_gap
from bossyk_sandbox.detector.transfer_eval import (
    assemble_eval_domain,
    build_cell_arrays,
    load_detector_scores,
    regenerate_bow_scores,
    verify_bow_regen_against_frozen,
)


def _row(
    row_id: str, scenario_id: str, prompt: str, action: str, *, is_violation: bool
) -> dict[str, object]:
    return {
        "row_id": row_id,
        "scenario_id": scenario_id,
        "prompt": prompt,
        "action": action,
        "is_violation": is_violation,
        "domain": "x",
    }


def _write_domain(
    data_root: Path,
    domain: str,
    *,
    n_train: int = 60,
    n_val: int = 10,
    n_test: int = 10,
    n_scenarios: int = 5,
) -> None:
    split_dir = data_root / domain / "v2"
    split_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(hash(domain) % (2**31))

    def make_rows(n: int, prefix: str) -> list[dict[str, object]]:
        rows = []
        for i in range(n):
            viol = bool(rng.integers(0, 2))
            scenario = f"{domain}-scenario-{i % n_scenarios}"
            action = (
                f"{domain} agent skips verification and cancels order {i}"
                if viol
                else f"{domain} agent verified id then cancels order {i}"
            )
            rows.append(
                _row(
                    f"{domain}:{prefix}:{i}",
                    scenario,
                    f"{domain} customer wants order {i} cancelled",
                    action,
                    is_violation=viol,
                )
            )
        return rows

    for split, n in (("train", n_train), ("val", n_val), ("test", n_test)):
        rows = make_rows(n, split)
        with (split_dir / f"{split}.jsonl").open("w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")


def test_regenerate_bow_scores_per_item_aggregates_match_hand_recomputed_auroc(
    tmp_path: Path,
) -> None:
    _write_domain(tmp_path, "alpha", n_train=100, n_test=30)
    _write_domain(tmp_path, "beta", n_train=100, n_test=30)
    domains = load_all_domains(("alpha", "beta"), data_root=tmp_path, corpus_version="v2")
    regen = regenerate_bow_scores(domains, seed=0)

    # Every eval row's row_id must appear in per_item with a finite score.
    cell = regen.cells["hashing"]["alpha"]["alpha"]
    per_item = regen.per_item["hashing"]["alpha"]["alpha"]
    assert cell.n == len(per_item)
    scores = np.array(list(per_item.values()))
    assert np.all(np.isfinite(scores))
    # The lexical signal ("skip verification" vs "verified id") is trivial,
    # so BoW should separate it well -- sanity check the regen pipeline
    # actually fits a discriminative probe, not a constant one.
    assert cell.auroc > 0.85


def test_verify_bow_regen_against_frozen_detects_match_and_mismatch(tmp_path: Path) -> None:
    _write_domain(tmp_path, "alpha")
    _write_domain(tmp_path, "beta")
    domains = load_all_domains(("alpha", "beta"), data_root=tmp_path, corpus_version="v2")
    regen = regenerate_bow_scores(domains, seed=0)

    frozen_matching = {
        "bow": {
            vec: {
                train: {ev: {"auroc": cell.auroc} for ev, cell in row.items()}
                for train, row in regen.cells[vec].items()
            }
            for vec in ("fixed_vocab", "hashing")
        }
    }
    assert verify_bow_regen_against_frozen(regen, frozen_matching) == []

    frozen_broken = json.loads(json.dumps(frozen_matching))
    frozen_broken["bow"]["hashing"]["alpha"]["alpha"]["auroc"] = 0.123456
    mismatches = verify_bow_regen_against_frozen(regen, frozen_broken)
    assert len(mismatches) == 1
    assert "hashing/alpha->alpha" in mismatches[0]


def _write_detector_scores(
    scores_root: Path,
    *,
    family: str,
    train_domain: str,
    seed: int,
    eval_domain: str,
    rows: list[dict[str, object]],
    perfect: bool,
) -> None:
    out_dir = scores_root / family / train_domain / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / f"{eval_domain}.jsonl").open("w") as f:
        for r in rows:
            label = bool(r["is_violation"])
            raw = (0.9 if label else 0.1) if perfect else 0.5
            f.write(
                json.dumps(
                    {
                        "row_id": r["row_id"],
                        "label": label,
                        "raw_score": raw,
                        "calibrated_score": raw,
                    }
                )
                + "\n"
            )


def test_load_detector_scores_reads_row_id_keyed_items(tmp_path: Path) -> None:
    _write_domain(tmp_path / "data", "alpha", n_train=20, n_val=5, n_test=5)
    domains = load_all_domains(("alpha",), data_root=tmp_path / "data", corpus_version="v2")
    scores_root = tmp_path / "scores"
    for seed in (0, 1, 2):
        _write_detector_scores(
            scores_root,
            family="deberta",
            train_domain="alpha",
            seed=seed,
            eval_domain="alpha",
            rows=domains["alpha"].test,
            perfect=True,
        )
    loaded = load_detector_scores(
        scores_root,
        families=("deberta",),
        train_domains=("alpha",),
        seeds=(0, 1, 2),
        eval_domains=("alpha",),
    )
    items = loaded["deberta"]["alpha"][0]["alpha"]
    assert len(items) == len(domains["alpha"].test)
    any_row_id = domains["alpha"].test[0]["row_id"]
    assert items[any_row_id].label == bool(domains["alpha"].test[0]["is_violation"])


def test_assemble_and_build_cell_arrays_end_to_end_with_bootstrap(tmp_path: Path) -> None:
    """End-to-end: synthetic corpus -> BoW regen (near-chance by
    construction, random labels) -> perfect-detector score files -> cell
    arrays -> hierarchical bootstrap gap should be strongly positive and
    exclude zero, since the detector is perfectly separated and BoW is
    chance."""
    data_root = tmp_path / "data"
    _write_domain(data_root, "retail", n_train=80, n_val=20, n_test=20, n_scenarios=8)
    _write_domain(data_root, "advice", n_train=80, n_val=20, n_test=20, n_scenarios=8)
    domains = load_all_domains(("retail", "advice"), data_root=data_root, corpus_version="v2")

    # Rewrite advice's full corpus rows with i.i.d. random labels (not
    # lexically determined) so BoW is near-chance against them, isolating
    # the detector's constructed perfection as the source of the gap.
    rng = np.random.default_rng(0)
    for r in domains["advice"].full:
        r["is_violation"] = bool(rng.integers(0, 2))
        r["action"] = f"advice agent processes decision {r['row_id']}"  # no lexical tell

    regen = regenerate_bow_scores(domains, seed=0)

    scores_root = tmp_path / "scores"
    for seed in (0, 1, 2):
        _write_detector_scores(
            scores_root,
            family="deberta",
            train_domain="retail",
            seed=seed,
            eval_domain="advice",
            rows=domains["advice"].full,
            perfect=True,
        )
    detector_scores = load_detector_scores(
        scores_root,
        families=("deberta",),
        train_domains=("retail",),
        seeds=(0, 1, 2),
        eval_domains=("advice",),
    )

    assembly = assemble_eval_domain(domains["advice"])
    cell = build_cell_arrays(
        train_domain="retail",
        eval_domain="advice",
        assembly=assembly,
        detector_by_seed={s: detector_scores["deberta"]["retail"][s]["advice"] for s in (0, 1, 2)},
        bow_scores_by_row_id=regen.per_item["fixed_vocab"]["retail"]["advice"],
    )

    result = bootstrap_composite_gap(
        [cell],
        {"advice": assembly.groups},
        n_resamples=500,
        seed=42,
        hierarchical=True,
    )
    assert result.point_gap > 0.3
    assert result.ci_excludes_zero is True


def test_build_cell_arrays_raises_on_label_mismatch(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _write_domain(data_root, "alpha", n_train=20, n_val=5, n_test=5)
    domains = load_all_domains(("alpha",), data_root=data_root, corpus_version="v2")
    assembly = assemble_eval_domain(domains["alpha"])

    corrupted_rows = [dict(r) for r in domains["alpha"].full]
    corrupted_rows[0]["is_violation"] = not corrupted_rows[0]["is_violation"]  # flip one label
    scores_root = tmp_path / "scores"
    _write_detector_scores(
        scores_root,
        family="deberta",
        train_domain="alpha",
        seed=0,
        eval_domain="alpha",
        rows=corrupted_rows,
        perfect=True,
    )
    loaded = load_detector_scores(
        scores_root,
        families=("deberta",),
        train_domains=("alpha",),
        seeds=(0,),
        eval_domains=("alpha",),
    )
    bow_scores = {r["row_id"]: 0.5 for r in domains["alpha"].full}
    with pytest.raises(ValueError, match="label mismatch"):
        build_cell_arrays(
            train_domain="alpha",
            eval_domain="alpha",
            assembly=assembly,
            detector_by_seed={0: loaded["deberta"]["alpha"][0]["alpha"]},
            bow_scores_by_row_id=bow_scores,
        )
