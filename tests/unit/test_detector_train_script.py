"""Hermetic tests for scripts/detector_train.py -- dry-run mode
(StubTrainBackend), tiny synthetic v2 corpora on disk, no network/torch."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import numpy as np

SCRIPT_PATH = Path(__file__).parent.parent.parent / "scripts" / "detector_train.py"


def _import() -> ModuleType:
    spec = importlib.util.spec_from_file_location("detector_train_script", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _write_domain(
    data_root: Path, domain: str, *, n_scenarios: int = 20, per_scenario: int = 5
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
    # crude split: last 20% scenarios -> test, next 15% -> val, rest -> train
    n_test = max(1, n_scenarios // 5)
    n_val = max(1, n_scenarios // 7)
    test_scenarios = {f"{domain}-scenario-{s}" for s in range(n_scenarios - n_test, n_scenarios)}
    val_scenarios = {
        f"{domain}-scenario-{s}" for s in range(n_scenarios - n_test - n_val, n_scenarios - n_test)
    }
    train_rows = [r for r in rows if r["scenario_id"] not in test_scenarios | val_scenarios]
    val_rows = [r for r in rows if r["scenario_id"] in val_scenarios]
    test_rows = [r for r in rows if r["scenario_id"] in test_scenarios]
    for split, split_rows in (("train", train_rows), ("val", val_rows), ("test", test_rows)):
        with (split_dir / f"{split}.jsonl").open("w") as f:
            for row in split_rows:
                f.write(json.dumps(row) + "\n")


def test_train_one_cell_dry_run_writes_checkpoint_and_returns_result(tmp_path: Path) -> None:
    m = _import()
    data_root = tmp_path / "data"
    _write_domain(data_root, "retail")
    ckpt_root = tmp_path / "checkpoints"
    backend = m.build_backend(dry_run=True)

    result = m.train_one_cell(
        family="deberta",
        train_domain="retail",
        seed=0,
        slice_seed=0,
        data_root=data_root,
        corpus_version="v2",
        checkpoint_root=ckpt_root,
        backend=backend,
        max_epochs=1,
        early_stopping_patience=1,
        per_device_batch_size=8,
        learning_rate=2e-5,
        lora_r=8,
        lora_alpha=16,
    )
    assert result.family == "deberta"
    assert result.train_domain == "retail"
    assert result.checkpoint_dir.exists()


def test_train_one_cell_rejects_unknown_family(tmp_path: Path) -> None:
    m = _import()
    data_root = tmp_path / "data"
    _write_domain(data_root, "retail")
    try:
        m.train_one_cell(
            family="not-a-family",
            train_domain="retail",
            seed=0,
            slice_seed=0,
            data_root=data_root,
            corpus_version="v2",
            checkpoint_root=tmp_path / "ckpt",
            backend=m.build_backend(dry_run=True),
            max_epochs=1,
            early_stopping_patience=1,
            per_device_batch_size=8,
            learning_rate=2e-5,
            lora_r=8,
            lora_alpha=16,
        )
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_append_manifest_row_writes_jsonl_with_cost_estimate(tmp_path: Path) -> None:
    m = _import()
    data_root = tmp_path / "data"
    _write_domain(data_root, "retail")
    result = m.train_one_cell(
        family="qwen_lora",
        train_domain="retail",
        seed=1,
        slice_seed=1,
        data_root=data_root,
        corpus_version="v2",
        checkpoint_root=tmp_path / "ckpt",
        backend=m.build_backend(dry_run=True),
        max_epochs=1,
        early_stopping_patience=1,
        per_device_batch_size=8,
        learning_rate=2e-5,
        lora_r=8,
        lora_alpha=16,
    )
    manifest_out = tmp_path / "manifest.jsonl"
    m.append_manifest_row(manifest_out, result)
    lines = manifest_out.read_text().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["family"] == "qwen_lora"
    assert row["seed"] == 1
    assert "cost_estimate_usd" in row
    assert row["cost_estimate_usd"] >= 0.0


def test_main_dry_run_end_to_end(tmp_path: Path) -> None:
    m = _import()
    data_root = tmp_path / "data"
    _write_domain(data_root, "airline")
    manifest_out = tmp_path / "manifest.jsonl"
    rc = m.main(
        [
            "--family",
            "deberta",
            "--train-domain",
            "airline",
            "--seed",
            "0",
            "--data-root",
            str(data_root),
            "--corpus-version",
            "v2",
            "--checkpoint-root",
            str(tmp_path / "ckpt"),
            "--manifest-out",
            str(manifest_out),
            "--max-epochs",
            "1",
            "--dry-run",
        ]
    )
    assert rc == 0
    lines = manifest_out.read_text().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["slice_seed"] == 0  # defaults to --seed
