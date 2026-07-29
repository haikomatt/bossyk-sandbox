"""Hermetic tests for scripts/text_baseline.py -- synthetic npz + decisions."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import numpy as np

SCRIPT_PATH = Path(__file__).parent.parent.parent / "scripts" / "text_baseline.py"


def _import() -> ModuleType:
    spec = importlib.util.spec_from_file_location("text_baseline_script", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_hashing_vectorize_deterministic_and_shaped() -> None:
    m = _import()
    a = m.hashing_vectorize(["cancel my order", "return policy please"], n_features=64)
    b = m.hashing_vectorize(["cancel my order", "return policy please"], n_features=64)
    assert a.shape == (2, 64)
    assert np.array_equal(a, b)  # md5 hashing is stable across calls/processes
    assert a.sum() > 0


def test_paired_diff_ci_detects_a_beats_b() -> None:
    m = _import()
    y = np.array([True] * 20 + [False] * 20)
    a = np.where(y, 0.9, 0.1) + 0.0  # perfect
    b = np.random.default_rng(0).uniform(0, 1, 40)  # noise
    lo, hi, p = m.paired_diff_ci(a, b, y, n_boot=500)
    assert lo > 0 and p > 0.9


def test_run_end_to_end(tmp_path: Path) -> None:
    m = _import()
    rng = np.random.default_rng(0)
    n = 60
    y = np.array([i < n // 2 for i in range(n)], dtype=bool)
    # activations separable on dim 0; text separable via distinct tokens per class
    X = rng.normal(0, 1, size=(n, 40))
    X[y, 0] += 3.0
    X[~y, 0] -= 3.0
    npz = tmp_path / "acts.npz"
    np.savez(
        npz,
        layers=np.array([0, 7], dtype=np.int64),
        is_violation=y,
        is_error=np.zeros(n, dtype=np.int8),
        step_ids=np.array([f"s{i}" for i in range(n)]),
        X_0=X.astype(np.float32),
        X_7=X.astype(np.float32),
    )
    rows = [
        {
            "step_id": f"s{i}",
            "prompt": ("cancel now" if y[i] else "return policy"),
            "is_violation": bool(y[i]),
        }
        for i in range(n)
    ]
    dec = tmp_path / "d.json"
    dec.write_text(json.dumps(rows))

    report = m.run(str(npz), str(dec), layers=[0, 7], n_components=10)
    assert report["n_items"] == n and report["n_violation"] == n // 2
    assert report["text_baseline_auroc"] > 0.9  # tokens separate the classes
    assert report["text_shuffled_auroc"] < 0.75  # control sane
    assert set(report["layers"]) == {"0", "7"}
    assert "beats_text" in report["layers"]["0"]
