"""Hermetic tests for scripts/detector_pod_train_all.py -- dry-run
(StubTrainBackend), tiny synthetic v2 corpora, no network/torch/pod.

Covers the three safety-rail behaviours the brief asks for: checkpointing
(resume skips already-manifested cells), the wall-clock/dollar budget
stopping the loop BEFORE starting a cell it can't finish, and a full
dry-run completing all cells when the budget is generous."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

SCRIPT_PATH = Path(__file__).parent.parent.parent / "scripts" / "detector_pod_train_all.py"


def _import() -> ModuleType:
    spec = importlib.util.spec_from_file_location("detector_pod_train_all_script", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _write_domain(
    data_root: Path, domain: str, *, n_scenarios: int = 12, per_scenario: int = 4
) -> None:
    split_dir = data_root / domain / "v2"
    split_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(hash(domain) % (2**31))
    rows = []
    for s in range(n_scenarios):
        viol = bool(rng.integers(0, 2))
        for i in range(per_scenario):
            rows.append(
                {
                    "row_id": f"{domain}:{s}:{i}",
                    "scenario_id": f"{domain}-scenario-{s}",
                    "prompt": f"{domain} customer wants order {s}-{i} cancelled",
                    "action": "skips verification and cancels"
                    if viol
                    else "verified id then cancels",
                    "is_violation": viol,
                }
            )
    n_test = max(1, n_scenarios // 4)
    test_scenarios = {f"{domain}-scenario-{s}" for s in range(n_scenarios - n_test, n_scenarios)}
    train_rows = [r for r in rows if r["scenario_id"] not in test_scenarios]
    test_rows = [r for r in rows if r["scenario_id"] in test_scenarios]
    for split, split_rows in (("train", train_rows), ("val", []), ("test", test_rows)):
        with (split_dir / f"{split}.jsonl").open("w") as f:
            for row in split_rows:
                f.write(json.dumps(row) + "\n")


def _run_kwargs(tmp_path: Path, data_root: Path, *, domains: tuple[str, ...]) -> dict[str, Any]:
    return dict(
        families=("deberta", "qwen_lora"),
        domains=domains,
        seeds=(0, 1),
        data_root=data_root,
        corpus_version="v2",
        checkpoint_root=tmp_path / "ckpt",
        manifest_out=tmp_path / "manifest.jsonl",
        scores_root=tmp_path / "scores",
        max_epochs=1,
        early_stopping_patience=1,
        per_device_batch_size=8,
        learning_rate=2e-5,
        lora_r=8,
        lora_alpha=16,
        dry_run=True,
    )


def test_run_all_dry_run_completes_all_cells_with_generous_budget(tmp_path: Path) -> None:
    m = _import()
    data_root = tmp_path / "data"
    _write_domain(data_root, "alpha")
    _write_domain(data_root, "beta")

    summary = m.run_all(**_run_kwargs(tmp_path, data_root, domains=("alpha", "beta")))

    assert summary["all_cells_complete"] is True
    assert summary["requested_cells"] == 2 * 2 * 2  # families x domains x seeds
    assert len(summary["ran_this_invocation"]) == 8
    assert summary["stopped_reason"] is None

    manifest_lines = (tmp_path / "manifest.jsonl").read_text().splitlines()
    assert len(manifest_lines) == 8

    # every trained cell produced score files for both eval domains
    for family, domain, seed in summary["ran_this_invocation"]:
        for eval_domain in ("alpha", "beta"):
            assert (
                tmp_path / "scores" / family / domain / f"seed{seed}" / f"{eval_domain}.jsonl"
            ).exists()


def test_run_all_deletes_checkpoints_after_scoring_by_default(tmp_path: Path) -> None:
    # Regression test for the real disk-exhaustion incident (2026-08-28
    # pod run): a 50GB container disk filled after 17/18 cells because
    # checkpoints were never cleaned up between cells. Scores + manifest
    # are what get downloaded -- checkpoints must not accumulate.
    m = _import()
    data_root = tmp_path / "data"
    _write_domain(data_root, "alpha")

    summary = m.run_all(**_run_kwargs(tmp_path, data_root, domains=("alpha",)))

    assert summary["all_cells_complete"] is True
    for family, domain, seed in summary["ran_this_invocation"]:
        cell_dir = tmp_path / "ckpt" / family / domain / f"seed{seed}"
        assert not cell_dir.exists()
    # scores survive the cleanup
    for family, domain, seed in summary["ran_this_invocation"]:
        assert (tmp_path / "scores" / family / domain / f"seed{seed}" / "alpha.jsonl").exists()


def test_run_all_keep_checkpoints_flag_preserves_them(tmp_path: Path) -> None:
    m = _import()
    data_root = tmp_path / "data"
    _write_domain(data_root, "alpha")

    kwargs = _run_kwargs(tmp_path, data_root, domains=("alpha",))
    kwargs["keep_checkpoints"] = True
    summary = m.run_all(**kwargs)

    assert summary["all_cells_complete"] is True
    for family, domain, seed in summary["ran_this_invocation"]:
        cell_dir = tmp_path / "ckpt" / family / domain / f"seed{seed}"
        assert cell_dir.exists()


def test_run_all_resumes_and_skips_already_manifested_cells(tmp_path: Path) -> None:
    m = _import()
    data_root = tmp_path / "data"
    _write_domain(data_root, "alpha")

    kwargs = _run_kwargs(tmp_path, data_root, domains=("alpha",))
    first = m.run_all(**kwargs)
    assert first["all_cells_complete"] is True
    n_manifest_after_first = len((tmp_path / "manifest.jsonl").read_text().splitlines())

    # Re-run with the same manifest_out: everything should be skipped.
    second = m.run_all(**kwargs)
    assert second["ran_this_invocation"] == []
    assert len(second["already_done_at_start"]) == 4  # 2 families x 1 domain x 2 seeds
    assert second["all_cells_complete"] is True
    n_manifest_after_second = len((tmp_path / "manifest.jsonl").read_text().splitlines())
    assert n_manifest_after_second == n_manifest_after_first  # no duplicate rows appended


def test_run_all_stops_before_dollar_cap_and_reports_why(tmp_path: Path) -> None:
    m = _import()
    data_root = tmp_path / "data"
    _write_domain(data_root, "alpha")

    kwargs = _run_kwargs(tmp_path, data_root, domains=("alpha",))
    kwargs["hard_cap_usd"] = 0.0001  # effectively zero budget
    kwargs["max_wall_seconds"] = 10**9
    summary = m.run_all(**kwargs)

    assert summary["ran_this_invocation"] == []
    assert summary["all_cells_complete"] is False
    assert summary["stopped_reason"] is not None
    assert "budget" in summary["stopped_reason"]


def test_run_all_stops_when_estimated_next_cell_wont_fit(tmp_path: Path) -> None:
    m = _import()
    data_root = tmp_path / "data"
    _write_domain(data_root, "alpha")

    kwargs = _run_kwargs(tmp_path, data_root, domains=("alpha",))
    # Budget is nonzero but tiny relative to a deliberately huge per-cell
    # time estimate -- the loop must refuse to start ANY cell rather than
    # start one it can't plausibly finish.
    kwargs["hard_cap_usd"] = 1000.0
    kwargs["max_wall_seconds"] = 5  # 5 seconds of wall budget
    kwargs["first_cell_estimate_seconds"] = 3600.0  # 1h estimate, way over budget
    summary = m.run_all(**kwargs)

    assert summary["ran_this_invocation"] == []
    assert "insufficient budget" in summary["stopped_reason"]


def test_main_dry_run_writes_summary(tmp_path: Path) -> None:
    m = _import()
    data_root = tmp_path / "data"
    _write_domain(data_root, "alpha")
    summary_out = tmp_path / "summary.json"

    rc = m.main(
        [
            "--families",
            "deberta",
            "--domains",
            "alpha",
            "--seeds",
            "0",
            "--data-root",
            str(data_root),
            "--corpus-version",
            "v2",
            "--checkpoint-root",
            str(tmp_path / "ckpt"),
            "--manifest-out",
            str(tmp_path / "manifest.jsonl"),
            "--scores-root",
            str(tmp_path / "scores"),
            "--summary-out",
            str(summary_out),
            "--max-epochs",
            "1",
            "--dry-run",
        ]
    )
    assert rc == 0
    summary = json.loads(summary_out.read_text())
    assert summary["all_cells_complete"] is True
    assert summary["requested_cells"] == 1
