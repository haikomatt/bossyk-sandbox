"""Hermetic tests for scripts/leadtime_probe.py (group-CV + per-offset analysis)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np

SCRIPT = Path(__file__).parent.parent.parent / "scripts" / "leadtime_probe.py"


def _import() -> ModuleType:
    spec = importlib.util.spec_from_file_location("leadtime_script", SCRIPT)
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


def test_group_oof_never_trains_and_tests_same_prompt() -> None:
    m = _import()
    # 6 prompts x 10 rollouts; label is a per-PROMPT constant separable on dim 0.
    # Group-CV must still recover it (prompt held out entirely), proving no leak.
    rng = np.random.default_rng(0)
    xs, ys, gs = [], [], []
    for p in range(6):
        viol = p < 3
        for _ in range(10):
            v = rng.normal(0, 1, 8)
            v[0] += 3.0 if viol else -3.0
            xs.append(v)
            ys.append(viol)
            gs.append(p)
    x = np.array(xs)
    y = np.array(ys)
    g = np.array(gs)
    oof = m.group_oof_scores(x, y, g, n_splits=3, n_components=4)
    from bossyk_sandbox.interp.correlate import auroc

    mask = ~np.isnan(oof)
    assert auroc(oof[mask].tolist(), y[mask].tolist()) > 0.9


def test_analyse_reports_per_offset_rows() -> None:
    m = _import()
    rng = np.random.default_rng(1)
    n, d = 40, 6
    y = np.array([i < n // 2 for i in range(n)])
    prompt_id = np.array([i % 8 for i in range(n)])  # 8 prompt groups
    offsets = np.array([-4, -2, -1])
    # signal present only at offset -1 (index 2); other offsets are noise
    X = rng.normal(0, 1, (n, 3, d))
    X[y, 2, 0] += 3.0
    X[~y, 2, 0] -= 3.0
    payload = {
        "offsets": offsets,
        "is_violation": y,
        "prompt_id": prompt_id,
        "text_so_far": np.array([["", "", "cancel" if y[i] else "look"] for i in range(n)]),
        "X_7": X.astype(np.float32),
    }
    out = m.analyse(payload, layer=7, n_splits=4)
    assert out["layer"] == 7
    assert [r["offset"] for r in out["offsets"]] == [-4, -2, -1]
    # offset -1 carries the signal -> higher residual AUROC than the noise offsets
    by_off = {r["offset"]: r["residual_auroc"] for r in out["offsets"]}
    assert by_off[-1] > 0.8
