"""Quality-check tooling for generated decision corpora
(phase-detector-training-step2-datagen.md Part A, item 3): independent
cross-split/cross-domain leakage scan, per-domain/per-split class-balance
report, a deterministic hand-audit sample export, and the provenance/honesty
manifest. Pure/offline -- no network, no LLM.

`leakage_scan` deliberately does NOT trust `corpus_assembly.stratified_group_split`
to have gotten grouping right -- it re-derives group membership from the
already-split output and reports any group key found in more than one split,
so a bug in the splitter (or a hand-edited split file) is still caught.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from bossyk_sandbox.interp.corpus_assembly import (
    Record,
    jaccard_similarity,
    near_duplicate_comparison_text,
    token_shingles,
)

# --- leakage scan ------------------------------------------------------


@dataclass(frozen=True)
class LeakedGroup:
    key: tuple[Any, ...]
    splits: list[str]


@dataclass
class LeakageReport:
    group_fields: tuple[str, ...]
    leaked_groups: list[LeakedGroup]

    @property
    def clean(self) -> bool:
        return not self.leaked_groups


def leakage_scan(
    splits: dict[str, list[Record]], *, group_fields: tuple[str, ...] = ("scenario_id",)
) -> LeakageReport:
    """For each field in `group_fields` independently, find any value that
    appears in more than one split -- e.g. the same `scenario_id` (or, when
    checked, the same `persona_id`) showing up in both train and val is a
    leak regardless of which OTHER field grouped it there. Rows with a
    `None` value for a field are exempt from that field's check (a missing
    key can't leak)."""
    group_to_splits: dict[tuple[Any, ...], set[str]] = {}
    for split_name, rows in splits.items():
        for row in rows:
            for field_name in group_fields:
                value = row.get(field_name)
                if value is None:
                    continue
                key = (value,)
                group_to_splits.setdefault(key, set()).add(split_name)

    leaked = [
        LeakedGroup(key=key, splits=sorted(present))
        for key, present in group_to_splits.items()
        if len(present) > 1
    ]
    leaked.sort(key=lambda g: g.key)
    return LeakageReport(group_fields=group_fields, leaked_groups=leaked)


# --- cross-domain near-duplicate scan --------------------------------------


@dataclass(frozen=True)
class CrossDomainPair:
    row_id_a: str
    row_id_b: str
    domain_a: str
    domain_b: str
    similarity: float


@dataclass
class CrossDomainDuplicateReport:
    n_pairs_checked: int
    pairs: list[CrossDomainPair]

    @property
    def rate(self) -> float:
        """Fraction of checked cross-domain pairs that cleared the
        similarity threshold. 0.0 (not NaN) when nothing was checked, so a
        caller can always report a number."""
        if self.n_pairs_checked == 0:
            return 0.0
        return len(self.pairs) / self.n_pairs_checked


def cross_domain_near_duplicate_scan(
    records: list[Record], *, threshold: float = 0.85, shingle_size: int = 5
) -> CrossDomainDuplicateReport:
    """O(n^2) across DIFFERENT domains only (same-domain near-dupes are
    `corpus_assembly.dedupe_near_duplicates`'s job, already run before this
    ever sees the corpus) -- a real leakage risk if two domains' prompt
    templates turn out too similar despite the lexical-overlap confound gate
    (`scripts/lexical_overlap_audit.py`) having passed at the vocabulary
    level; this checks actual generated prompt TEXT (via
    `near_duplicate_comparison_text` -- the shared per-domain system-policy
    boilerplate is stripped first, same as the within-domain dedupe, so
    cross-domain comparisons aren't dominated by two DIFFERENT domains'
    unrelated policy text either), a different and complementary signal."""
    by_domain: dict[str, list[Record]] = {}
    for r in records:
        by_domain.setdefault(str(r["domain"]), []).append(r)
    domains = sorted(by_domain)

    shingle_cache = {
        id(r): token_shingles(near_duplicate_comparison_text(str(r["prompt"])), shingle_size)
        for r in records
    }

    pairs: list[CrossDomainPair] = []
    n_checked = 0
    for i, domain_a in enumerate(domains):
        for domain_b in domains[i + 1 :]:
            for row_a in by_domain[domain_a]:
                for row_b in by_domain[domain_b]:
                    n_checked += 1
                    sim = jaccard_similarity(shingle_cache[id(row_a)], shingle_cache[id(row_b)])
                    if sim >= threshold:
                        pairs.append(
                            CrossDomainPair(
                                row_id_a=str(row_a["row_id"]),
                                row_id_b=str(row_b["row_id"]),
                                domain_a=domain_a,
                                domain_b=domain_b,
                                similarity=sim,
                            )
                        )
    pairs.sort(key=lambda p: (-p.similarity, p.row_id_a, p.row_id_b))
    return CrossDomainDuplicateReport(n_pairs_checked=n_checked, pairs=pairs)


# --- class balance -----------------------------------------------------


def class_balance_report(
    splits: dict[str, list[Record]],
) -> dict[str, dict[str, dict[str, float | int]]]:
    """domain -> split -> {n, violations, violation_rate}. Domains and split
    names are both read from the data, never hardcoded, so a new domain or
    split name shows up automatically."""
    report: dict[str, dict[str, dict[str, float | int]]] = {}
    for split_name, rows in splits.items():
        by_domain: dict[str, list[Record]] = {}
        for row in rows:
            by_domain.setdefault(str(row["domain"]), []).append(row)
        for domain, domain_rows in by_domain.items():
            n = len(domain_rows)
            violations = sum(1 for r in domain_rows if r["is_violation"])
            report.setdefault(domain, {})[split_name] = {
                "n": n,
                "violations": violations,
                "violation_rate": (violations / n) if n else 0.0,
            }
    return report


# --- hand-audit sample ---------------------------------------------------


def export_hand_audit_sample(records: list[Record], *, n: int = 50, seed: int = 0) -> list[Record]:
    """Deterministic seeded sample of up to `n` records (fewer if the corpus
    is smaller), for a human label-audit pass. Sampling without replacement,
    order-independent given a fixed seed (sorted by row_id before sampling)
    so the result doesn't depend on the corpus's on-disk row order."""
    ordered = sorted(records, key=lambda r: str(r.get("row_id", "")))
    if len(ordered) <= n:
        return ordered
    return random.Random(seed).sample(ordered, n)


# --- provenance / honesty manifest ------------------------------------------


def build_manifest(
    *,
    domain: str,
    generator_model: str,
    serving_path: str,
    seed: int,
    counts: dict[str, int],
    class_balance: dict[str, float],
    cap_hit: str | None,
    near_dup_rate_within_domain: float,
    generated_at: str,
) -> dict[str, Any]:
    """The per-corpus provenance/honesty record (bossyk convention): every
    number a downstream reader needs to judge whether this corpus is fit for
    training, in one place. `cap_was_hit` is a derived, unambiguous bool
    alongside the raw `cap_hit` reason string, so a reader doesn't have to
    remember that `None` means "no cap hit"."""
    return {
        "domain": domain,
        "generator_model": generator_model,
        "serving_path": serving_path,
        "seed": seed,
        "generated_at": generated_at,
        "counts": counts,
        "class_balance": class_balance,
        "cap_hit": cap_hit,
        "cap_was_hit": cap_hit is not None,
        "near_dup_rate_within_domain": near_dup_rate_within_domain,
    }
