"""Hermetic tests for scripts/detector_score.py -- dry-run mode
(StubTrainBackend), tiny synthetic v2 corpora, no network/torch."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import numpy as np

SCRIPT_PATH = Path(__file__).parent.parent.parent / "scripts" / "detector_score.py"


def _import() -> ModuleType:
    spec = importlib.util.spec_from_file_location("detector_score_script", SCRIPT_PATH)
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


def test_score_cell_writes_one_file_per_eval_domain(tmp_path: Path) -> None:
    m = _import()
    data_root = tmp_path / "data"
    _write_domain(data_root, "alpha")
    _write_domain(data_root, "beta")
    out_root = tmp_path / "scores"
    backend = m.build_backend(dry_run=True)

    written = m.score_cell(
        family="deberta",
        train_domain="alpha",
        seed=0,
        checkpoint_dir=Path("/nonexistent-stub-checkpoint"),
        temperature=1.5,
        data_root=data_root,
        corpus_version="v2",
        domains=("alpha", "beta"),
        out_root=out_root,
        backend=backend,
    )
    assert set(written) == {"alpha", "beta"}
    for path in written.values():
        assert path.exists()


def test_score_cell_in_domain_uses_test_split_ood_uses_full_corpus(tmp_path: Path) -> None:
    m = _import()
    data_root = tmp_path / "data"
    _write_domain(data_root, "alpha", n_scenarios=12, per_scenario=4)
    _write_domain(data_root, "beta", n_scenarios=12, per_scenario=4)
    out_root = tmp_path / "scores"
    backend = m.build_backend(dry_run=True)

    written = m.score_cell(
        family="deberta",
        train_domain="alpha",
        seed=0,
        checkpoint_dir=Path("/nonexistent-stub-checkpoint"),
        temperature=1.0,
        data_root=data_root,
        corpus_version="v2",
        domains=("alpha", "beta"),
        out_root=out_root,
        backend=backend,
    )
    in_domain_lines = written["alpha"].read_text().splitlines()
    ood_lines = written["beta"].read_text().splitlines()
    assert all(json.loads(line)["eval_set"] == "test_split" for line in in_domain_lines)
    assert all(json.loads(line)["eval_set"] == "full_unique_corpus" for line in ood_lines)
    # beta's full corpus (train+val+test) is bigger than alpha's test split alone
    assert len(ood_lines) > len(in_domain_lines)


def test_score_cell_item_has_raw_and_calibrated_scores(tmp_path: Path) -> None:
    m = _import()
    data_root = tmp_path / "data"
    _write_domain(data_root, "alpha")
    out_root = tmp_path / "scores"
    backend = m.build_backend(dry_run=True)

    written = m.score_cell(
        family="deberta",
        train_domain="alpha",
        seed=0,
        checkpoint_dir=Path("/nonexistent-stub-checkpoint"),
        temperature=2.0,
        data_root=data_root,
        corpus_version="v2",
        domains=("alpha",),
        out_root=out_root,
        backend=backend,
    )
    lines = written["alpha"].read_text().splitlines()
    assert lines
    for line in lines:
        item = json.loads(line)
        for key in (
            "row_id",
            "label",
            "raw_logit",
            "raw_score",
            "temperature",
            "calibrated_score",
        ):
            assert key in item
        assert 0.0 <= item["raw_score"] <= 1.0
        assert 0.0 <= item["calibrated_score"] <= 1.0
        assert item["temperature"] == 2.0


def test_score_cell_rejects_unknown_family(tmp_path: Path) -> None:
    m = _import()
    data_root = tmp_path / "data"
    _write_domain(data_root, "alpha")
    try:
        m.score_cell(
            family="not-a-family",
            train_domain="alpha",
            seed=0,
            checkpoint_dir=Path("/x"),
            temperature=1.0,
            data_root=data_root,
            corpus_version="v2",
            domains=("alpha",),
            out_root=tmp_path / "scores",
            backend=m.build_backend(dry_run=True),
        )
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_main_dry_run_end_to_end(tmp_path: Path) -> None:
    m = _import()
    data_root = tmp_path / "data"
    _write_domain(data_root, "alpha")
    _write_domain(data_root, "beta")
    out_root = tmp_path / "scores"
    rc = m.main(
        [
            "--family",
            "deberta",
            "--train-domain",
            "alpha",
            "--seed",
            "0",
            "--temperature",
            "1.1",
            "--checkpoint-dir",
            "/nonexistent",
            "--data-root",
            str(data_root),
            "--corpus-version",
            "v2",
            "--domains",
            "alpha,beta",
            "--out-root",
            str(out_root),
            "--dry-run",
        ]
    )
    assert rc == 0
    assert (out_root / "deberta" / "alpha" / "seed0" / "alpha.jsonl").exists()
    assert (out_root / "deberta" / "alpha" / "seed0" / "beta.jsonl").exists()


def test_main_derives_checkpoint_dir_from_checkpoint_root(tmp_path: Path) -> None:
    m = _import()
    data_root = tmp_path / "data"
    _write_domain(data_root, "alpha")
    ckpt_root = tmp_path / "ckpt"
    (ckpt_root / "deberta" / "alpha" / "seed0" / "final").mkdir(parents=True)
    rc = m.main(
        [
            "--family",
            "deberta",
            "--train-domain",
            "alpha",
            "--seed",
            "0",
            "--temperature",
            "1.0",
            "--checkpoint-root",
            str(ckpt_root),
            "--data-root",
            str(data_root),
            "--domains",
            "alpha",
            "--out-root",
            str(tmp_path / "scores"),
            "--dry-run",
        ]
    )
    assert rc == 0
