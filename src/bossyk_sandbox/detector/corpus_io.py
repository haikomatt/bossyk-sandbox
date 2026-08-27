"""Shared v2-corpus loading for the detector-training scripts
(`scripts/detector_train.py`, `scripts/detector_score.py`,
`scripts/detector_pod_train_all.py`). Extends, rather than duplicates,
`scripts/detector_baselines.py`'s `DomainData`/`load_jsonl` pattern -- that
script is the frozen-baselines script (do not touch, per house rules); this
module is the reusable core the new training/scoring scripts import instead
of re-writing the same JSONL-loading logic a third time.

Read-only: never writes to `probes/detector/data/**`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from bossyk_sandbox.interp.corpus_assembly import Record, corpus_data_dir

DEFAULT_DATA_ROOT = Path(__file__).parent.parent.parent.parent / "probes" / "detector" / "data"
DEFAULT_DOMAINS: tuple[str, ...] = ("retail", "airline", "advice-eligibility")


def load_jsonl(path: Path) -> list[Record]:
    if not path.exists():
        return []
    rows: list[Record] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


@dataclass(frozen=True)
class DomainSplits:
    """One domain's v2 train/val/test rows, plus the derived `full` corpus
    (train+val+test) -- the OOD eval set per Amendment 2's corrected
    definition (`a-fine-tuned-violation-detector-transfers-cross-domain.md`:
    "the OOD evaluation set for a train-A/test-B cell is B's FULL unique
    corpus")."""

    domain: str
    train: list[Record]
    val: list[Record]
    test: list[Record]

    @property
    def full(self) -> list[Record]:
        return self.train + self.val + self.test

    def eval_rows(self, eval_domain: DomainSplits) -> list[Record]:
        """The eval set for a cell scoring `eval_domain`'s data: its own
        test split if this domain IS eval_domain (in-domain), else its full
        corpus (OOD)."""
        if eval_domain.domain == self.domain:
            return eval_domain.test
        return eval_domain.full


def load_domain_splits(
    domain: str, *, data_root: Path = DEFAULT_DATA_ROOT, corpus_version: str = "v2"
) -> DomainSplits:
    split_dir = corpus_data_dir(data_root, domain, corpus_version)
    return DomainSplits(
        domain=domain,
        train=load_jsonl(split_dir / "train.jsonl"),
        val=load_jsonl(split_dir / "val.jsonl"),
        test=load_jsonl(split_dir / "test.jsonl"),
    )


def load_all_domains(
    domains: tuple[str, ...] = DEFAULT_DOMAINS,
    *,
    data_root: Path = DEFAULT_DATA_ROOT,
    corpus_version: str = "v2",
) -> dict[str, DomainSplits]:
    return {
        d: load_domain_splits(d, data_root=data_root, corpus_version=corpus_version)
        for d in domains
    }
