from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

# The exact top-level metadata fields every per-domain H1 JSON must agree on
# before their domain blocks can be merged into one cross-domain report.
# Order matters here -- it is reused as the merged output's key order.
_METADATA_FIELDS = ("adversary_model", "guardrail_backend", "intensity")


class CrossDomainMergeError(ValueError):
    """Raised when the per-domain inputs cannot be merged as given -- either
    their metadata disagrees (different adversary/guardrail/intensity, which
    would make "cross-domain" comparison meaningless) or two inputs claim the
    same domain."""


def _load_per_domain_h1(path: Path | str) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(Path(path).read_text())
    return data


def merge_crossdomain_h1(paths: Sequence[Path | str]) -> dict[str, Any]:
    """Pure merge over already-persisted per-domain H1 JSON files (as
    produced by `scripts/h1_bench.py`) into the cross-domain H1 summary
    shape committed at `docs/bench_output/phase2b_crossdomain_h1.json`.

    For each input: takes the `canonical_strength` strength's
    `h1_overall`/`h1_by_class`/`h1_by_boundary` blocks verbatim (whatever
    keys they carry -- older artifacts have no `n_refused`/`n_error`, newer
    ones do; both pass through untouched) and adds a `strength_sweep`
    mapping every swept strength to its `h1_overall.rate`.

    `generated_from` records the paths exactly as given (not resolved),
    matching the committed file's relative-path convention.

    Raises `CrossDomainMergeError` if two inputs disagree on
    adversary_model/guardrail_backend/intensity (comparing rates across
    domains attacked by different adversaries or scored by different
    guardrail configs would not be a meaningful cross-domain read) or if
    two inputs claim the same domain."""
    if not paths:
        raise CrossDomainMergeError("at least one per-domain H1 JSON file is required")

    generated_from = [str(path) for path in paths]
    metadata: dict[str, Any] | None = None
    canonical_strength: str | None = None
    domains: dict[str, Any] = {}

    for original_path, path in zip(generated_from, paths, strict=True):
        data = _load_per_domain_h1(path)
        config = data["config"]
        domain_name = config["domain"]
        this_canonical_strength = data["canonical_strength"]

        this_metadata = {field: config[field] for field in _METADATA_FIELDS}
        if metadata is None:
            metadata = this_metadata
            canonical_strength = this_canonical_strength
        else:
            if this_metadata != metadata:
                raise CrossDomainMergeError(
                    f"{original_path!r} has metadata {this_metadata!r}, which "
                    f"disagrees with the earlier input(s)' {metadata!r} -- "
                    "cross-domain H1 rates are only comparable when every "
                    "domain was attacked/guardrailed under the same config"
                )
            if this_canonical_strength != canonical_strength:
                raise CrossDomainMergeError(
                    f"{original_path!r} has canonical_strength "
                    f"{this_canonical_strength!r}, which disagrees with the "
                    f"earlier input(s)' {canonical_strength!r}"
                )

        if domain_name in domains:
            raise CrossDomainMergeError(
                f"domain {domain_name!r} appears more than once across the given "
                f"input files (duplicate at {original_path!r})"
            )

        strengths = data["strengths"]
        canonical_block = strengths[this_canonical_strength]

        domains[domain_name] = {
            "h1_overall": dict(canonical_block["h1_overall"]),
            "h1_by_class": {
                attack_class: dict(block)
                for attack_class, block in canonical_block["h1_by_class"].items()
            },
            "h1_by_boundary": {
                boundary: dict(block)
                for boundary, block in canonical_block["h1_by_boundary"].items()
            },
            "strength_sweep": {
                strength: block["h1_overall"]["rate"] for strength, block in strengths.items()
            },
        }

    assert metadata is not None
    assert canonical_strength is not None
    return {
        "generated_from": generated_from,
        **metadata,
        "canonical_strength": canonical_strength,
        "domains": domains,
    }
