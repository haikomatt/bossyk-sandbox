#!/usr/bin/env python
"""Frozen pre-training baselines, free half (phase-detector-training.md
build-order step 3): bag-of-words + logreg (both vectorisers) and the
trivial majority-class baseline, evaluated over the pre-registered 3x3
transfer matrix. Pure/offline -- no network, no LLM.

Per domain, fits ONLY on that domain's v2 TRAIN split (hyperparameters
chosen by CV within train). Evaluates every (train_domain, eval_domain)
cell:
  - in-domain (train_domain == eval_domain): eval_domain's TEST split.
  - OOD (train_domain != eval_domain): eval_domain's FULL unique corpus
    (train+val+test) -- the corrected OOD definition from
    `a-fine-tuned-violation-detector-transfers-cross-domain.md` Amendment 2
    ("the OOD evaluation set for a train-A/test-B cell is B's full unique
    corpus, not B's test split -- B is never trained on by that detector, so
    restricting to B's test split discards evidence for no leakage benefit").

The "decision text" a row contributes is its `prompt` (system policy +
conversation so far) followed by the `action` the agent took -- the policy
label is about the ACTION, so both vectorisers see it, matching what the
zero-shot judge baseline is also given (`scripts/detector_judge_baseline.py`).

Usage:
    uv run python scripts/detector_baselines.py \\
        --domains retail,airline,advice-eligibility --corpus-version v2 \\
        --out probes/detector/results/baselines_bow_majority.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from bossyk_sandbox.detector.bow_baseline import (
    DEFAULT_L2_GRID,
    POWER_GATE,
    CellResult,
    MajorityBaselineResult,
    fit_domain_probe,
    majority_class_baseline,
)
from bossyk_sandbox.detector.bow_baseline import evaluate_cell as _evaluate_bow_cell
from bossyk_sandbox.detector.text_features import (
    FixedVocab,
    fit_fixed_vocab,
    fixed_vocab_vectorize,
    hashing_vectorize,
)
from bossyk_sandbox.interp.corpus_assembly import corpus_data_dir

Array = NDArray[np.float64]

DATA_ROOT = Path(__file__).parent.parent / "probes" / "detector" / "data"
DEFAULT_DOMAINS = ("retail", "airline", "advice-eligibility")
FIXED_VOCAB_MAX_FEATURES = 4096
FIXED_VOCAB_MIN_DF = 2
HASHING_N_FEATURES = 4096


def decision_text(row: dict[str, Any]) -> str:
    """The text a BoW/judge baseline sees for one decision: the context
    prompt, then the action the agent took. The policy label is about the
    ACTION, not just the request, so the action must be visible."""
    return f"{row['prompt']}\n\nACTION:\n{row['action']}"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


class DomainData:
    """One domain's v2 train/val/test rows, plus the derived arrays this
    script needs repeatedly: decision texts and boolean labels for the train
    split, the test split (in-domain eval), and the full unique corpus
    (train+val+test -- the OOD eval set per Amendment 2)."""

    def __init__(self, domain: str, data_root: Path, corpus_version: str) -> None:
        split_dir = corpus_data_dir(data_root, domain, corpus_version)
        self.domain = domain
        self.train_rows = load_jsonl(split_dir / "train.jsonl")
        self.val_rows = load_jsonl(split_dir / "val.jsonl")
        self.test_rows = load_jsonl(split_dir / "test.jsonl")
        self.full_rows = self.train_rows + self.val_rows + self.test_rows

        self.train_texts = [decision_text(r) for r in self.train_rows]
        self.train_y: NDArray[np.bool_] = np.array(
            [bool(r["is_violation"]) for r in self.train_rows], dtype=bool
        )
        self.test_texts = [decision_text(r) for r in self.test_rows]
        self.test_y: NDArray[np.bool_] = np.array(
            [bool(r["is_violation"]) for r in self.test_rows], dtype=bool
        )
        self.full_texts = [decision_text(r) for r in self.full_rows]
        self.full_y: NDArray[np.bool_] = np.array(
            [bool(r["is_violation"]) for r in self.full_rows], dtype=bool
        )

    def eval_texts_and_y(self, eval_domain: DomainData) -> tuple[list[str], NDArray[np.bool_]]:
        """The eval set for a cell scoring `eval_domain`'s data: its own test
        split if this IS eval_domain (in-domain), else its full corpus (OOD)."""
        if eval_domain.domain == self.domain:
            return eval_domain.test_texts, eval_domain.test_y
        return eval_domain.full_texts, eval_domain.full_y


def _cell_to_dict(cell: CellResult, *, in_domain: bool) -> dict[str, Any]:
    return {
        "auroc": cell.auroc,
        "ece": cell.ece,
        "n": cell.n,
        "n_pos": cell.n_pos,
        "underpowered": cell.underpowered,
        "in_domain": in_domain,
        "eval_set": "test_split" if in_domain else "full_unique_corpus",
    }


def run_bow_matrix(
    domains: dict[str, DomainData],
    *,
    vectorizer: str,
    seed: int = 0,
    l2_grid: tuple[float, ...] = DEFAULT_L2_GRID,
    power_gate: int = POWER_GATE,
) -> dict[str, Any]:
    """Fit one BoW+logreg probe per train domain (train split only, L2 chosen
    by CV within train), evaluate it against every domain's eval set. Returns
    `{train_domain: {eval_domain: cell_dict, ..., "chosen_l2": float}}`."""
    if vectorizer not in ("fixed_vocab", "hashing"):
        raise ValueError(f"unknown vectorizer: {vectorizer!r}")
    matrix: dict[str, Any] = {}
    for train_domain, train_data in domains.items():
        if vectorizer == "fixed_vocab":
            vocab: FixedVocab = fit_fixed_vocab(
                train_data.train_texts,
                max_features=FIXED_VOCAB_MAX_FEATURES,
                min_df=FIXED_VOCAB_MIN_DF,
            )
            x_train = fixed_vocab_vectorize(vocab, train_data.train_texts)

            def transform(texts: list[str], _vocab: FixedVocab = vocab) -> Array:
                return fixed_vocab_vectorize(_vocab, texts)

        else:
            x_train = hashing_vectorize(train_data.train_texts, n_features=HASHING_N_FEATURES)

            def transform(texts: list[str]) -> Array:
                return hashing_vectorize(texts, n_features=HASHING_N_FEATURES)

        probe, chosen_l2 = fit_domain_probe(x_train, train_data.train_y, l2_grid=l2_grid, seed=seed)

        row: dict[str, Any] = {"chosen_l2": chosen_l2}
        for eval_domain, eval_data in domains.items():
            texts, y = train_data.eval_texts_and_y(eval_data)
            x_eval = transform(texts)
            cell = _evaluate_bow_cell(probe, x_eval, y, power_gate=power_gate)
            row[eval_domain] = _cell_to_dict(cell, in_domain=(eval_domain == train_domain))
        matrix[train_domain] = row
    return matrix


def run_majority_matrix(domains: dict[str, DomainData]) -> dict[str, Any]:
    """The trivial majority-class baseline over the same 3x3 matrix, for
    completeness -- fit (i.e. pick the majority label) on the train domain's
    train split, evaluate on every eval domain's eval set."""
    matrix: dict[str, Any] = {}
    for train_domain, train_data in domains.items():
        row: dict[str, Any] = {}
        for eval_domain, eval_data in domains.items():
            _texts, y = train_data.eval_texts_and_y(eval_data)
            result: MajorityBaselineResult = majority_class_baseline(train_data.train_y, y)
            row[eval_domain] = {
                "majority_label": result.majority_label,
                "auroc": float("nan"),  # constant score -> AUROC undefined, not a spurious 0.5
                "accuracy": result.accuracy,
                "n": result.n,
                "n_pos": result.n_pos,
                "in_domain": eval_domain == train_domain,
                "eval_set": "test_split" if eval_domain == train_domain else "full_unique_corpus",
            }
        matrix[train_domain] = row
    return matrix


def domain_summary(domains: dict[str, DomainData]) -> dict[str, Any]:
    return {
        name: {
            "train_n": len(d.train_rows),
            "train_pos": int(d.train_y.sum()),
            "test_n": len(d.test_rows),
            "test_pos": int(d.test_y.sum()),
            "full_n": len(d.full_rows),
            "full_pos": int(d.full_y.sum()),
        }
        for name, d in domains.items()
    }


def run(
    domains_list: list[str],
    *,
    data_root: Path = DATA_ROOT,
    corpus_version: str = "v2",
    seed: int = 0,
) -> dict[str, Any]:
    domains = {d: DomainData(d, data_root, corpus_version) for d in domains_list}
    return {
        "config": {
            "seed": seed,
            "corpus_version": corpus_version,
            "l2_grid": list(DEFAULT_L2_GRID),
            "power_gate": POWER_GATE,
            "fixed_vocab": {"max_features": FIXED_VOCAB_MAX_FEATURES, "min_df": FIXED_VOCAB_MIN_DF},
            "hashing": {"n_features": HASHING_N_FEATURES},
        },
        "domains": domain_summary(domains),
        "bow_fixed_vocab": run_bow_matrix(domains, vectorizer="fixed_vocab", seed=seed),
        "bow_hashing": run_bow_matrix(domains, vectorizer="hashing", seed=seed),
        "majority_class": run_majority_matrix(domains),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--domains", default=",".join(DEFAULT_DOMAINS))
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--corpus-version", default="v2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    domains_list = [d.strip() for d in args.domains.split(",") if d.strip()]
    report = run(
        domains_list, data_root=args.data_root, corpus_version=args.corpus_version, seed=args.seed
    )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))
        print(f"wrote {args.out}", file=sys.stderr)

    for vec_name in ("bow_fixed_vocab", "bow_hashing"):
        print(f"\n=== {vec_name} ===")
        for train_domain, row in report[vec_name].items():
            for eval_domain, cell in row.items():
                if eval_domain == "chosen_l2":
                    continue
                flag = " UNDERPOWERED" if cell["underpowered"] else ""
                print(
                    f"  train={train_domain:20s} eval={eval_domain:20s} "
                    f"auroc={cell['auroc']:.3f} ece={cell['ece']:.3f} "
                    f"n_pos={cell['n_pos']}{flag}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
