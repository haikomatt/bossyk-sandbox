"""Assembly + stratified group-split for generated decision corpora
(phase-detector-training-step2-datagen.md Part A, item 2).

Pure/offline: reads the JSONL rows `datagen_driver.append_task_result`
writes, dedupes them (exact + near-duplicate context), and splits
train/val/test GROUPED so the same underlying scenario (and, when the data
actually carries one, the same persona/entity) never appears in two splits.

Grouping key is CONFIGURABLE (`group_fields`), defaulting to `("scenario_id",)`
-- always real, read off every row -- and extended to also include
`persona_id` only when at least one row in the corpus actually carries a
non-null one (advice-eligibility today; retail/airline don't, per
`datagen_driver`'s docstring, so grouping rests on scenario_id alone there).
Nothing is invented: `default_group_fields` never adds a key the data doesn't
have.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

Record = dict[str, Any]


def load_decisions_jsonl(path: Path) -> list[Record]:
    """Load a checkpoint/decisions JSONL file, dropping the `empty=True`
    task-sentinel rows `datagen_driver.append_task_result` writes for a task
    that produced zero decisions (they carry no `prompt`/`is_violation` and
    aren't corpus content)."""
    if not path.exists():
        return []
    rows: list[Record] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if row.get("empty"):
            continue
        rows.append(row)
    return rows


# --- dedupe ------------------------------------------------------------


def dedupe_exact(records: list[Record]) -> tuple[list[Record], int]:
    """Drops records whose (domain, prompt) exact text has already been
    seen, keeping the first occurrence. Returns (kept, n_dropped)."""
    seen: set[tuple[Any, str]] = set()
    kept: list[Record] = []
    dropped = 0
    for r in records:
        key = (r.get("domain"), r["prompt"])
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        kept.append(r)
    return kept, dropped


def token_shingles(text: str, k: int) -> set[str]:
    tokens = text.lower().split()
    if len(tokens) == 0:
        return set()
    if len(tokens) < k:
        return {" ".join(tokens)}
    return {" ".join(tokens[i : i + k]) for i in range(len(tokens) - k + 1)}


def jaccard_similarity(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def dedupe_near_duplicates(
    records: list[Record],
    *,
    threshold: float = 0.9,
    shingle_size: int = 5,
    bucket_fields: tuple[str, ...] = ("domain", "is_violation"),
) -> tuple[list[Record], int]:
    """Drops records whose prompt is a near-duplicate (token-shingle Jaccard
    >= `threshold`) of an already-kept record, keeping the first occurrence
    of each near-duplicate cluster (greedy, order-preserving). Comparisons
    are bounded to records sharing the same `bucket_fields` (default: same
    domain + same label) so this stays roughly O(n) per bucket rather than
    O(n^2) over the whole corpus. `threshold` and `shingle_size` are both
    tunable -- the plan doesn't pin a specific number, so the QC report is
    what makes a chosen threshold auditable (see
    `corpus_qc.cross_domain_near_duplicate_scan`, the cross-domain counterpart
    of this same shingle-Jaccard machinery).
    Returns (kept, n_dropped)."""
    buckets: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for i, r in enumerate(records):
        buckets[tuple(r.get(f) for f in bucket_fields)].append(i)

    shingle_sets = [token_shingles(r["prompt"], shingle_size) for r in records]
    kept_idx: list[int] = []
    dropped = 0
    for idxs in buckets.values():
        kept_in_bucket: list[int] = []
        for i in idxs:
            is_dup = any(
                jaccard_similarity(shingle_sets[i], shingle_sets[j]) >= threshold
                for j in kept_in_bucket
            )
            if is_dup:
                dropped += 1
            else:
                kept_in_bucket.append(i)
        kept_idx.extend(kept_in_bucket)
    kept_idx.sort()
    return [records[i] for i in kept_idx], dropped


@dataclass
class DedupeReport:
    n_input: int
    n_exact_dropped: int
    n_near_dropped: int
    n_output: int


def dedupe(
    records: list[Record], *, near_dup_threshold: float = 0.9, shingle_size: int = 5
) -> tuple[list[Record], DedupeReport]:
    """Exact dedupe, then near-duplicate dedupe on what's left. Order
    matters: exact dupes are cheap/unambiguous to drop first, shrinking the
    near-dup comparison set."""
    n_input = len(records)
    after_exact, n_exact = dedupe_exact(records)
    after_near, n_near = dedupe_near_duplicates(
        after_exact, threshold=near_dup_threshold, shingle_size=shingle_size
    )
    return after_near, DedupeReport(
        n_input=n_input, n_exact_dropped=n_exact, n_near_dropped=n_near, n_output=len(after_near)
    )


# --- grouping + stratified split ----------------------------------------


def default_group_fields(records: list[Record]) -> tuple[str, ...]:
    """`scenario_id` always (real, never invented). `persona_id` is added
    ONLY when at least one record in this corpus actually carries a non-null
    one -- see module docstring; this never synthesizes a persona field the
    data doesn't have."""
    if any(r.get("persona_id") is not None for r in records):
        return ("scenario_id", "persona_id")
    return ("scenario_id",)


def _union_find_groups(
    records: list[Record], group_fields: tuple[str, ...]
) -> dict[int, list[int]]:
    """Any two records sharing a non-null value on ANY of `group_fields` land
    in the same group (union-find over the fields) -- e.g. records A/B share
    a scenario_id, B/C share a persona_id -> A, B, and C are all one group.
    This is what makes "grouped by persona/scenario" airtight even when the
    two keys overlap only partially."""
    parent = list(range(len(records)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for field_name in group_fields:
        first_seen: dict[Any, int] = {}
        for i, r in enumerate(records):
            value = r.get(field_name)
            if value is None:
                continue
            if value in first_seen:
                union(i, first_seen[value])
            else:
                first_seen[value] = i

    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(len(records)):
        groups[find(i)].append(i)
    return groups


@dataclass
class SplitReport:
    group_fields: tuple[str, ...]
    n_groups: int
    counts: dict[str, dict[str, int]]  # split -> {"n": int, "violations": int}


def stratified_group_split(
    records: list[Record],
    *,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 0,
    group_fields: tuple[str, ...] | None = None,
) -> tuple[dict[str, list[Record]], SplitReport]:
    """Splits `records` into train/val/test, whole-group at a time (see
    `_union_find_groups`), via a deterministic seeded shuffle + greedy
    label-stratified bin-packing: each group goes to whichever split is
    currently furthest below its (label-aware) target share. Since whole
    groups move together, exact target fractions aren't hit precisely --
    the greedy choice minimises the drift, and `SplitReport.counts` reports
    what was actually achieved so a caller can check it's close enough."""
    if not 0.0 <= val_frac < 1.0 or not 0.0 <= test_frac < 1.0 or val_frac + test_frac >= 1.0:
        raise ValueError(f"val_frac={val_frac} + test_frac={test_frac} must be < 1.0 (each >= 0)")
    fields = group_fields if group_fields is not None else default_group_fields(records)

    if not records:
        report = SplitReport(
            group_fields=fields,
            n_groups=0,
            counts={s: {"n": 0, "violations": 0} for s in ("train", "val", "test")},
        )
        return {"train": [], "val": [], "test": []}, report

    groups = _union_find_groups(records, fields)
    keys = list(groups)
    random.Random(seed).shuffle(keys)

    totals = {"pos": 0, "neg": 0}
    group_labels: dict[int, tuple[int, int]] = {}
    for key in keys:
        idxs = groups[key]
        pos = sum(1 for i in idxs if records[i]["is_violation"])
        neg = len(idxs) - pos
        group_labels[key] = (pos, neg)
        totals["pos"] += pos
        totals["neg"] += neg

    split_fracs = {"val": val_frac, "test": test_frac, "train": 1.0 - val_frac - test_frac}
    targets = {
        s: {"pos": totals["pos"] * f, "neg": totals["neg"] * f} for s, f in split_fracs.items()
    }
    current = {s: {"pos": 0, "neg": 0} for s in split_fracs}
    assigned: dict[str, list[int]] = {"train": [], "val": [], "test": []}

    for key in keys:
        pos, neg = group_labels[key]

        def deficit(split: str, _pos: int = pos, _neg: int = neg) -> float:
            return (targets[split]["pos"] - current[split]["pos"]) + (
                targets[split]["neg"] - current[split]["neg"]
            )

        best = max(split_fracs, key=deficit)
        assigned[best].extend(groups[key])
        current[best]["pos"] += pos
        current[best]["neg"] += neg

    splits = {name: [records[i] for i in sorted(idxs)] for name, idxs in assigned.items()}
    counts = {
        name: {
            "n": len(rows),
            "violations": sum(1 for r in rows if r["is_violation"]),
        }
        for name, rows in splits.items()
    }
    report = SplitReport(group_fields=fields, n_groups=len(keys), counts=counts)
    return splits, report
