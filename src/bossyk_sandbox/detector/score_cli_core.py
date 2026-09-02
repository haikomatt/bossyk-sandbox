"""Shared core for `scripts/detector_score.py` (one checkpoint) and
`scripts/detector_pod_train_all.py` (trains + scores all 18 cells) --
extends rather than duplicates the scoring logic, per house "extend before
create" rule.

Given ONE trained (family, train_domain, seed) checkpoint, emits PER-ITEM
scores for every eval domain -- in-domain = that domain's v2 test split, OOD
= the tested domain's full unique corpus (Amendment 2's corrected OOD
definition) -- to `<out_root>/<family>/<train_domain>/seed<k>/<eval_domain>.jsonl`.
Both raw (T=1) and calibrated (Amendment-3 temperature) scores are saved.
Emits SCORES ONLY -- no paired-gap CIs, no verdict language.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from bossyk_sandbox.detector.corpus_io import load_all_domains
from bossyk_sandbox.detector.temperature import apply_temperature
from bossyk_sandbox.detector.train_backend import FAMILIES, TrainBackend


def score_cell(
    *,
    family: str,
    train_domain: str,
    seed: int,
    checkpoint_dir: Path,
    temperature: float,
    data_root: Path,
    corpus_version: str,
    domains: tuple[str, ...],
    out_root: Path,
    backend: TrainBackend,
) -> dict[str, Path]:
    """Score `checkpoint_dir` against every `eval_domain` in `domains`;
    writes one JSONL file per eval domain and returns `{eval_domain: path}`.
    """
    if family not in FAMILIES:
        raise ValueError(f"unknown family: {family!r} (expected one of {FAMILIES})")
    all_domains = load_all_domains(domains, data_root=data_root, corpus_version=corpus_version)
    if train_domain not in all_domains:
        raise ValueError(f"train_domain {train_domain!r} not in loaded domains {domains}")
    train_splits = all_domains[train_domain]

    out_dir = out_root / family / train_domain / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    written: dict[str, Path] = {}
    for eval_domain, eval_splits in all_domains.items():
        rows = train_splits.eval_rows(eval_splits)
        in_domain = eval_domain == train_domain
        logits = (
            backend.score(family=family, checkpoint_dir=checkpoint_dir, rows=rows) if rows else []
        )

        out_path = out_dir / f"{eval_domain}.jsonl"
        with out_path.open("w") as f:
            for row, logit in zip(rows, logits, strict=True):
                raw_score = float(apply_temperature(np.array([logit]), 1.0)[0])
                calibrated_score = float(apply_temperature(np.array([logit]), temperature)[0])
                item: dict[str, Any] = {
                    "row_id": row.get("row_id"),
                    "family": family,
                    "train_domain": train_domain,
                    "seed": seed,
                    "eval_domain": eval_domain,
                    "in_domain": in_domain,
                    "eval_set": "test_split" if in_domain else "full_unique_corpus",
                    "label": bool(row["is_violation"]),
                    "raw_logit": float(logit),
                    "raw_score": raw_score,
                    "temperature": temperature,
                    "calibrated_score": calibrated_score,
                }
                f.write(json.dumps(item) + "\n")
        written[eval_domain] = out_path
    return written
