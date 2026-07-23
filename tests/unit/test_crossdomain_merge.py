from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from bossyk_sandbox.scoring.crossdomain_merge import CrossDomainMergeError, merge_crossdomain_h1

REPO_ROOT = Path(__file__).parent.parent.parent
BENCH_OUTPUT = REPO_ROOT / "docs" / "bench_output"


def _h1_block(
    n_attempts: int, n_bypassed: int, *, extra: dict[str, Any] | None = None
) -> dict[str, Any]:
    block: dict[str, Any] = {
        "n_attempts": n_attempts,
        "n_bypassed": n_bypassed,
        "rate": n_bypassed / n_attempts,
        "wilson_ci95": [0.0, 1.0],
    }
    if extra:
        block.update(extra)
    return block


def _per_domain_payload(
    *,
    domain: str,
    adversary_model: str = "accounts/fireworks/models/deepseek-v4-pro",
    guardrail_backend: str = "model-backed",
    intensity: str = "aggressive",
    canonical_strength: str = "leaky",
    class_name: str,
    boundary_name: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    def strength_block(bypassed: int) -> dict[str, Any]:
        return {
            "h1_overall": _h1_block(40, bypassed, extra=extra),
            "h1_by_class": {class_name: _h1_block(40, bypassed, extra=extra)},
            "h1_by_boundary": {boundary_name: _h1_block(40, bypassed, extra=extra)},
            "n_crossings": bypassed,
        }

    return {
        "generated_at": "2026-07-01T00:00:00Z",
        "mode": "smoke",
        "config": {
            "domain": domain,
            "adversary_model": adversary_model,
            "guardrail_backend": guardrail_backend,
            "intensity": intensity,
            "budget": 10,
            "grid": {
                "attack_classes": [class_name],
                "boundaries": [boundary_name],
                "n_cells": 1,
                "total_attempts": 40,
            },
        },
        "strengths": {
            "leaky": strength_block(10),
            "moderate": strength_block(8),
            "strict": strength_block(6),
        },
        "canonical_strength": canonical_strength,
        "regression_probe_count": 1,
        "generated_attempts": [],
    }


def _write(tmp_path: Path, name: str, payload: dict[str, Any]) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload))
    return path


def test_merge_combines_two_domains_under_their_canonical_strength(tmp_path: Path) -> None:
    airline_path = _write(
        tmp_path,
        "airline.json",
        _per_domain_payload(
            domain="airline", class_name="jailbreak", boundary_name="cancel_without_lookup"
        ),
    )
    retail_path = _write(
        tmp_path,
        "retail.json",
        _per_domain_payload(
            domain="retail", class_name="jailbreak", boundary_name="cancel_without_auth"
        ),
    )

    merged = merge_crossdomain_h1([str(airline_path), str(retail_path)])

    assert merged["generated_from"] == [str(airline_path), str(retail_path)]
    assert merged["adversary_model"] == "accounts/fireworks/models/deepseek-v4-pro"
    assert merged["guardrail_backend"] == "model-backed"
    assert merged["intensity"] == "aggressive"
    assert merged["canonical_strength"] == "leaky"
    assert set(merged["domains"]) == {"airline", "retail"}

    airline = merged["domains"]["airline"]
    assert airline["h1_overall"] == _h1_block(40, 10)
    assert airline["h1_by_class"] == {"jailbreak": _h1_block(40, 10)}
    assert airline["h1_by_boundary"] == {"cancel_without_lookup": _h1_block(40, 10)}
    assert airline["strength_sweep"] == {"leaky": 0.25, "moderate": 0.2, "strict": 0.15}


def test_generated_from_records_paths_exactly_as_given_not_resolved(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "airline.json",
        _per_domain_payload(
            domain="airline", class_name="jailbreak", boundary_name="cancel_without_lookup"
        ),
    )
    relative_style = f"./{path.name}"
    # Give it a path string that isn't the absolute tmp_path form -- prove
    # the merge doesn't silently normalize/resolve it.
    import os

    old_cwd = Path.cwd()
    try:
        os.chdir(path.parent)
        merged = merge_crossdomain_h1([relative_style])
    finally:
        os.chdir(old_cwd)

    assert merged["generated_from"] == [relative_style]


def test_older_generation_blocks_without_n_refused_pass_through_unchanged(tmp_path: Path) -> None:
    # Older H1 artifacts have no n_refused/n_error keys -- the merge must
    # not invent them.
    path = _write(
        tmp_path,
        "airline.json",
        _per_domain_payload(
            domain="airline", class_name="jailbreak", boundary_name="cancel_without_lookup"
        ),
    )

    merged = merge_crossdomain_h1([str(path)])

    overall = merged["domains"]["airline"]["h1_overall"]
    assert "n_refused" not in overall
    assert "n_error" not in overall


def test_newer_generation_carries_n_refused_and_n_error_through(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "airline.json",
        _per_domain_payload(
            domain="airline",
            class_name="jailbreak",
            boundary_name="cancel_without_lookup",
            extra={"n_refused": 3, "n_error": 1, "n_scored": 36, "refusal_rate": 0.075},
        ),
    )

    merged = merge_crossdomain_h1([str(path)])

    overall = merged["domains"]["airline"]["h1_overall"]
    assert overall["n_refused"] == 3
    assert overall["n_error"] == 1
    assert overall["n_scored"] == 36
    assert overall["refusal_rate"] == 0.075

    by_class = merged["domains"]["airline"]["h1_by_class"]["jailbreak"]
    assert by_class["n_refused"] == 3


def test_mismatched_adversary_model_across_inputs_raises_loudly(tmp_path: Path) -> None:
    airline_path = _write(
        tmp_path,
        "airline.json",
        _per_domain_payload(
            domain="airline", class_name="jailbreak", boundary_name="cancel_without_lookup"
        ),
    )
    retail_path = _write(
        tmp_path,
        "retail.json",
        _per_domain_payload(
            domain="retail",
            class_name="jailbreak",
            boundary_name="cancel_without_auth",
            adversary_model="claude-fable-5",
        ),
    )

    with pytest.raises(CrossDomainMergeError, match="metadata"):
        merge_crossdomain_h1([str(airline_path), str(retail_path)])


def test_mismatched_canonical_strength_across_inputs_raises_loudly(tmp_path: Path) -> None:
    airline_path = _write(
        tmp_path,
        "airline.json",
        _per_domain_payload(
            domain="airline", class_name="jailbreak", boundary_name="cancel_without_lookup"
        ),
    )
    retail_path = _write(
        tmp_path,
        "retail.json",
        _per_domain_payload(
            domain="retail",
            class_name="jailbreak",
            boundary_name="cancel_without_auth",
            canonical_strength="strict",
        ),
    )

    with pytest.raises(CrossDomainMergeError, match="canonical_strength"):
        merge_crossdomain_h1([str(airline_path), str(retail_path)])


def test_duplicate_domain_across_inputs_raises_loudly(tmp_path: Path) -> None:
    first = _write(
        tmp_path,
        "airline1.json",
        _per_domain_payload(
            domain="airline", class_name="jailbreak", boundary_name="cancel_without_lookup"
        ),
    )
    second = _write(
        tmp_path,
        "airline2.json",
        _per_domain_payload(
            domain="airline", class_name="jailbreak", boundary_name="cancel_without_lookup"
        ),
    )

    with pytest.raises(CrossDomainMergeError, match="airline"):
        merge_crossdomain_h1([str(first), str(second)])


def test_no_input_paths_raises_loudly() -> None:
    with pytest.raises(CrossDomainMergeError):
        merge_crossdomain_h1([])


# --- Acceptance check: merge over the real committed inputs reproduces the
# real committed cross-domain file (remediation item 3 / code-review
# finding 17) ------------------------------------------------------------


def test_merging_the_committed_per_domain_inputs_matches_the_committed_output() -> None:
    committed = json.loads((BENCH_OUTPUT / "phase2b_crossdomain_h1.json").read_text())
    generated_from = committed["generated_from"]

    merged = merge_crossdomain_h1(generated_from)

    assert merged == committed
