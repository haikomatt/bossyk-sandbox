"""Data-assembly layer for step 5 (`scripts/detector_verdict.py`): loads the
v2 corpora, regenerates the frozen BoW baseline's per-item scores (with a
built-in check against `probes/detector/results/baselines_frozen.json`),
loads the 54 committed detector per-item score files, and assembles the
row-id-aligned arrays `hierarchical_bootstrap` needs.

Reuses rather than duplicates:
- `bossyk_sandbox.detector.corpus_io` for the v2 train/val/test/full splits
  (the same reader `scripts/detector_score.py` used to produce the eval sets
  the committed score files score against).
- `scripts/detector_baselines.py` (the FROZEN baselines script -- never
  modified) for `decision_text` and the fixed-vocab/hashing vectoriser
  constants, imported dynamically via `importlib.util`, the same pattern
  `tests/unit/test_detector_baselines_script.py` already uses to exercise
  that script without turning it into an importable package.
- `bossyk_sandbox.detector.bow_baseline` (already a package module, not the
  script) for `fit_domain_probe` -- the actual probe-fitting code path.
- `bossyk_sandbox.interp.correlate.auroc` / `detector.calibration.
  expected_calibration_error` for every REPORTED point statistic, so the
  regenerated numbers are computed with the exact same functions that
  produced the frozen table (not a numerically-close reimplementation).
"""

from __future__ import annotations

import importlib.util
import json
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
from numpy.typing import NDArray

from bossyk_sandbox.detector.bow_baseline import DEFAULT_L2_GRID, fit_domain_probe
from bossyk_sandbox.detector.calibration import expected_calibration_error
from bossyk_sandbox.detector.corpus_io import DomainSplits, load_all_domains
from bossyk_sandbox.detector.hierarchical_bootstrap import CellArrays, DomainGroups
from bossyk_sandbox.interp.correlate import auroc
from bossyk_sandbox.interp.probe import probe_scores

Array = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

REPO_ROOT = Path(__file__).parent.parent.parent.parent
BASELINES_SCRIPT_PATH = REPO_ROOT / "scripts" / "detector_baselines.py"
FAMILIES: tuple[str, ...] = ("deberta", "qwen_lora")
SEEDS: tuple[int, ...] = (0, 1, 2)
VECTORIZERS: tuple[str, ...] = ("fixed_vocab", "hashing")
POWER_GATE = 150
BOW_REGEN_TOLERANCE = 1e-9


def _load_baselines_script() -> ModuleType:
    """Dynamically import `scripts/detector_baselines.py` (the frozen
    baselines script, never modified) so its `decision_text` and vectoriser
    constants are reused rather than re-typed a second time. Same pattern as
    `tests/unit/test_detector_baselines_script.py`."""
    spec = importlib.util.spec_from_file_location(
        "detector_baselines_frozen", BASELINES_SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class BowCell:
    auroc: float
    ece: float
    n: int
    n_pos: int
    underpowered: bool


@dataclass(frozen=True)
class BowRegen:
    """Regenerated BoW per-item scores and per-cell aggregates, for both
    vectorisers, over the full 3x3 matrix. `per_item[vectorizer][train_domain]
    [eval_domain]` is `{row_id: raw_score}`; `cells[...]` is the matching
    aggregate (computed from those SAME per-item scores via the canonical
    `interp.correlate.auroc` / `calibration.expected_calibration_error`)."""

    chosen_l2: dict[str, dict[str, float]]  # vectorizer -> train_domain -> l2
    per_item: dict[
        str, dict[str, dict[str, dict[str, float]]]
    ]  # vec -> train -> eval -> {row_id: score}
    cells: dict[str, dict[str, dict[str, BowCell]]]  # vec -> train -> eval -> BowCell


def _decision_texts_and_labels(
    rows: list[dict[str, Any]], decision_text: Any
) -> tuple[list[str], BoolArray, list[str]]:
    texts = [decision_text(r) for r in rows]
    y = np.array([bool(r["is_violation"]) for r in rows], dtype=bool)
    row_ids = [str(r["row_id"]) for r in rows]
    return texts, y, row_ids


def regenerate_bow_scores(
    domains: dict[str, DomainSplits], *, seed: int = 0, l2_grid: tuple[float, ...] = DEFAULT_L2_GRID
) -> BowRegen:
    """Refit the fixed-vocab and hashing BoW+logreg probes exactly as
    `scripts/detector_baselines.py::run_bow_matrix` does (same functions,
    same seed, same train-only fit), but additionally RETAIN per-item scores
    keyed by `row_id` -- the frozen table only stored aggregates, and the
    paired bootstrap gap needs per-item pairing against the detector's own
    per-item scores."""
    m = _load_baselines_script()
    chosen_l2: dict[str, dict[str, float]] = {"fixed_vocab": {}, "hashing": {}}
    per_item: dict[str, dict[str, dict[str, dict[str, float]]]] = {"fixed_vocab": {}, "hashing": {}}
    cells: dict[str, dict[str, dict[str, BowCell]]] = {"fixed_vocab": {}, "hashing": {}}

    for train_domain, train_splits in domains.items():
        train_texts, train_y, _ = _decision_texts_and_labels(train_splits.train, m.decision_text)

        for vec in VECTORIZERS:
            transform: Callable[[list[str]], Array]
            if vec == "fixed_vocab":
                vocab = m.fit_fixed_vocab(
                    train_texts,
                    max_features=m.FIXED_VOCAB_MAX_FEATURES,
                    min_df=m.FIXED_VOCAB_MIN_DF,
                )
                x_train = m.fixed_vocab_vectorize(vocab, train_texts)
                transform = partial(m.fixed_vocab_vectorize, vocab)
            else:
                x_train = m.hashing_vectorize(train_texts, n_features=m.HASHING_N_FEATURES)
                transform = partial(m.hashing_vectorize, n_features=m.HASHING_N_FEATURES)

            probe, l2 = fit_domain_probe(x_train, train_y, l2_grid=l2_grid, seed=seed)
            chosen_l2[vec][train_domain] = l2
            per_item[vec][train_domain] = {}
            cells[vec][train_domain] = {}

            for eval_domain, eval_splits in domains.items():
                in_domain = eval_domain == train_domain
                eval_rows = eval_splits.test if in_domain else eval_splits.full
                texts, y, row_ids = _decision_texts_and_labels(eval_rows, m.decision_text)
                x_eval = transform(texts)
                scores = probe_scores(probe, x_eval)
                per_item[vec][train_domain][eval_domain] = dict(
                    zip(row_ids, scores.tolist(), strict=True)
                )
                cell_auroc = auroc(scores.tolist(), y.tolist())
                cells[vec][train_domain][eval_domain] = BowCell(
                    auroc=cell_auroc,
                    ece=expected_calibration_error(scores, y),
                    n=int(len(y)),
                    n_pos=int(y.sum()),
                    underpowered=int(y.sum()) < POWER_GATE,
                )
    return BowRegen(chosen_l2=chosen_l2, per_item=per_item, cells=cells)


def verify_bow_regen_against_frozen(
    regen: BowRegen, frozen: dict[str, Any], *, tolerance: float = BOW_REGEN_TOLERANCE
) -> list[str]:
    """Compare every regenerated cell AUROC against `baselines_frozen.json`'s
    `bow.<vectorizer>.<train>.<eval>.auroc`. Returns the list of mismatches
    (empty if all match within `tolerance`) -- the caller aborts the script
    if this is non-empty (the brief's MANDATORY built-in check)."""
    mismatches: list[str] = []
    for vec in VECTORIZERS:
        frozen_vec = frozen["bow"][vec]
        for train_domain, row in regen.cells[vec].items():
            for eval_domain, cell in row.items():
                frozen_auroc = frozen_vec[train_domain][eval_domain]["auroc"]
                diff = abs(cell.auroc - frozen_auroc)
                if not (diff <= tolerance):
                    mismatches.append(
                        f"{vec}/{train_domain}->{eval_domain}: regen={cell.auroc!r} "
                        f"frozen={frozen_auroc!r} diff={diff!r} > tol={tolerance!r}"
                    )
    return mismatches


@dataclass(frozen=True)
class DetectorItem:
    label: bool
    raw_score: float
    calibrated_score: float


def load_detector_scores(
    scores_root: Path,
    *,
    families: tuple[str, ...] = FAMILIES,
    train_domains: tuple[str, ...],
    seeds: tuple[int, ...] = SEEDS,
    eval_domains: tuple[str, ...],
) -> dict[str, dict[str, dict[int, dict[str, dict[str, DetectorItem]]]]]:
    """`result[family][train_domain][seed][eval_domain] = {row_id: DetectorItem}`,
    read from the committed `probes/detector/results/scores/**` files
    (never written to)."""
    result: dict[str, dict[str, dict[int, dict[str, dict[str, DetectorItem]]]]] = {}
    for family in families:
        result[family] = {}
        for train_domain in train_domains:
            result[family][train_domain] = {}
            for seed in seeds:
                result[family][train_domain][seed] = {}
                for eval_domain in eval_domains:
                    path = (
                        scores_root / family / train_domain / f"seed{seed}" / f"{eval_domain}.jsonl"
                    )
                    items: dict[str, DetectorItem] = {}
                    for line in path.read_text().splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        row = json.loads(line)
                        items[str(row["row_id"])] = DetectorItem(
                            label=bool(row["label"]),
                            raw_score=float(row["raw_score"]),
                            calibrated_score=float(row["calibrated_score"]),
                        )
                    result[family][train_domain][seed][eval_domain] = items
    return result


@dataclass(frozen=True)
class EvalDomainAssembly:
    """One eval domain's canonical row order (its FULL unique corpus, per
    Amendment 2's OOD definition -- every cell this module builds is
    cross-domain), the resulting `DomainGroups` for the hierarchical
    bootstrap, and the aligned label array every cell scoring this domain
    must share."""

    row_ids: list[str]
    labels: BoolArray
    groups: DomainGroups


def assemble_eval_domain(domain_splits: DomainSplits) -> EvalDomainAssembly:
    rows = domain_splits.full
    row_ids = [str(r["row_id"]) for r in rows]
    labels = np.array([bool(r["is_violation"]) for r in rows], dtype=bool)
    scenario_indices: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        scenario_indices.setdefault(str(r["scenario_id"]), []).append(i)
    groups = DomainGroups(
        domain=domain_splits.domain, n=len(rows), scenario_indices=scenario_indices
    )
    return EvalDomainAssembly(row_ids=row_ids, labels=labels, groups=groups)


def build_cell_arrays(
    *,
    train_domain: str,
    eval_domain: str,
    assembly: EvalDomainAssembly,
    detector_by_seed: dict[int, dict[str, DetectorItem]],
    bow_scores_by_row_id: dict[str, float],
) -> CellArrays:
    """Align one (train_domain, eval_domain) cell's detector (per seed, raw
    score) and BoW (raw score) arrays to `assembly`'s canonical row order,
    cross-checking every item's label against `assembly.labels` (the
    corpus's own `is_violation` field) so a label mismatch between the score
    files and the corpus aborts loudly rather than silently misaligning."""
    n = len(assembly.row_ids)
    seed_scores: dict[int, Array] = {}
    for seed, items in detector_by_seed.items():
        arr = np.empty(n, dtype=np.float64)
        for i, row_id in enumerate(assembly.row_ids):
            item = items[row_id]
            if item.label != bool(assembly.labels[i]):
                raise ValueError(
                    f"label mismatch for {row_id} in {train_domain}->{eval_domain} seed{seed}: "
                    f"score file says {item.label}, corpus says {bool(assembly.labels[i])}"
                )
            arr[i] = item.raw_score
        seed_scores[seed] = arr

    bow_arr = np.array(
        [bow_scores_by_row_id[row_id] for row_id in assembly.row_ids], dtype=np.float64
    )

    return CellArrays(
        train_domain=train_domain,
        eval_domain=eval_domain,
        seed_scores=seed_scores,
        bow_scores=bow_arr,
        labels=assembly.labels,
    )


def load_domains(
    domains: tuple[str, ...], *, data_root: Path, corpus_version: str = "v2"
) -> dict[str, DomainSplits]:
    return load_all_domains(domains, data_root=data_root, corpus_version=corpus_version)
