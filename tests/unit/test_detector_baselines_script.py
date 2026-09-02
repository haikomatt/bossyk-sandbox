"""Hermetic tests for scripts/detector_baselines.py -- tiny synthetic v2
corpora on disk, no network."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import numpy as np

SCRIPT_PATH = Path(__file__).parent.parent.parent / "scripts" / "detector_baselines.py"


def _import() -> ModuleType:
    spec = importlib.util.spec_from_file_location("detector_baselines_script", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _row(row_id: str, prompt: str, action: str, *, is_violation: bool) -> dict[str, object]:
    return {
        "row_id": row_id,
        "prompt": prompt,
        "action": action,
        "is_violation": is_violation,
        "domain": "x",
    }


def _write_domain(
    data_root: Path, domain: str, *, n_train: int = 40, n_val: int = 10, n_test: int = 10
) -> None:
    """Write a tiny, linearly-separable-by-vocabulary v2 corpus: violation
    rows mention 'skip verification', compliant rows mention 'verified id'."""
    split_dir = data_root / domain / "v2"
    split_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(hash(domain) % (2**31))

    def make_rows(n: int, prefix: str) -> list[dict[str, object]]:
        rows = []
        for i in range(n):
            viol = bool(rng.integers(0, 2))
            action = (
                f"{domain} agent skips verification and cancels order {i}"
                if viol
                else f"{domain} agent verified id then cancels order {i}"
            )
            rows.append(
                _row(
                    f"{domain}:{prefix}:{i}",
                    f"{domain} customer wants order {i} cancelled",
                    action,
                    is_violation=viol,
                )
            )
        return rows

    for split, n in (("train", n_train), ("val", n_val), ("test", n_test)):
        rows = make_rows(n, split)
        with (split_dir / f"{split}.jsonl").open("w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")


def test_decision_text_includes_prompt_and_action() -> None:
    m = _import()
    text = m.decision_text({"prompt": "context here", "action": "cancel(W1)"})
    assert "context here" in text
    assert "cancel(W1)" in text


def test_run_produces_full_3x3_matrix_for_two_domains(tmp_path: Path) -> None:
    m = _import()
    _write_domain(tmp_path, "alpha")
    _write_domain(tmp_path, "beta")

    report = m.run(["alpha", "beta"], data_root=tmp_path, corpus_version="v2", seed=0)

    assert set(report["domains"]) == {"alpha", "beta"}
    for vec in ("bow_fixed_vocab", "bow_hashing"):
        matrix = report[vec]
        assert set(matrix) == {"alpha", "beta"}
        for train_domain, row in matrix.items():
            assert set(row) == {"alpha", "beta", "chosen_l2"}
            for eval_domain in ("alpha", "beta"):
                cell = row[eval_domain]
                assert cell["in_domain"] == (eval_domain == train_domain)
                expected_eval_set = (
                    "test_split" if eval_domain == train_domain else "full_unique_corpus"
                )
                assert cell["eval_set"] == expected_eval_set
                assert cell["n"] > 0
                assert 0 <= cell["n_pos"] <= cell["n"]


def test_bow_probe_separates_vocab_that_determines_the_label(tmp_path: Path) -> None:
    m = _import()
    _write_domain(tmp_path, "alpha", n_train=200, n_test=60)
    report = m.run(["alpha"], data_root=tmp_path, corpus_version="v2", seed=0)
    cell = report["bow_hashing"]["alpha"]["alpha"]
    # "skip verification" vs "verified id" is a trivial lexical signal -> BoW
    # should separate it well.
    assert cell["auroc"] > 0.9


def test_ood_eval_uses_full_corpus_not_just_test_split(tmp_path: Path) -> None:
    m = _import()
    _write_domain(tmp_path, "alpha", n_train=20, n_val=5, n_test=5)
    _write_domain(tmp_path, "beta", n_train=20, n_val=5, n_test=5)
    report = m.run(["alpha", "beta"], data_root=tmp_path, corpus_version="v2", seed=0)
    ood_cell = report["bow_hashing"]["alpha"]["beta"]
    assert ood_cell["n"] == 30  # beta's full unique corpus: 20 + 5 + 5
    in_domain_cell = report["bow_hashing"]["beta"]["beta"]
    assert in_domain_cell["n"] == 5  # beta's test split only


def test_majority_matrix_reports_accuracy_and_nan_auroc(tmp_path: Path) -> None:
    m = _import()
    _write_domain(tmp_path, "alpha")
    _write_domain(tmp_path, "beta")
    report = m.run(["alpha", "beta"], data_root=tmp_path, corpus_version="v2", seed=0)
    cell = report["majority_class"]["alpha"]["beta"]
    assert np.isnan(cell["auroc"])
    assert 0.0 <= cell["accuracy"] <= 1.0


def test_underpowered_flag_set_when_below_power_gate(tmp_path: Path) -> None:
    m = _import()
    _write_domain(tmp_path, "alpha", n_train=10, n_val=2, n_test=2)
    report = m.run(["alpha"], data_root=tmp_path, corpus_version="v2", seed=0)
    cell = report["bow_hashing"]["alpha"]["alpha"]
    assert cell["underpowered"] is True  # far fewer than 150 positives


def test_main_writes_json_report(tmp_path: Path) -> None:
    m = _import()
    _write_domain(tmp_path, "alpha")
    _write_domain(tmp_path, "beta")
    out = tmp_path / "report.json"
    rc = m.main(
        [
            "--domains",
            "alpha,beta",
            "--data-root",
            str(tmp_path),
            "--corpus-version",
            "v2",
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    written = json.loads(out.read_text())
    assert "bow_fixed_vocab" in written and "bow_hashing" in written and "majority_class" in written
