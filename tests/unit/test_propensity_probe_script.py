"""Hermetic tests for scripts/propensity_probe.py (H1a, the caveated secondary in
the H1 runbook): does the DETERMINISTIC pre-generation residual (offset 0,
identical across a prompt's rollouts) predict the per-prompt violation RATE,
against a bag-of-words baseline on the prompt text? Leave-one-prompt-out ridge,
Spearman on the held-out predictions. n = number of prompts (26), so the test
detects only a large effect and the script must say so."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

SCRIPT = Path(__file__).parent.parent.parent / "scripts" / "propensity_probe.py"


def _import() -> ModuleType:
    spec = importlib.util.spec_from_file_location("propensity_probe_script", SCRIPT)
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


def _payload(
    *, signal_in_residual: bool, n_prompts: int = 30, k: int = 12, d: int = 8
) -> dict[str, Any]:
    rng = np.random.default_rng(0)
    rate = rng.uniform(0.0, 1.0, n_prompts)
    prompt_id = np.repeat(np.arange(n_prompts), k)
    y = rng.uniform(0, 1, n_prompts * k) < rate[prompt_id]
    base = rng.normal(0, 1, (n_prompts, d))
    if signal_in_residual:
        base[:, 0] = 4.0 * rate + rng.normal(0, 0.1, n_prompts)
    X0 = base[prompt_id]  # identical across a prompt's rollouts, as at offset 0
    X = np.full((n_prompts * k, 2, d), np.nan)
    X[:, 0, :] = X0
    return {
        "offsets": np.array([0, 1]),
        "is_violation": y,
        "called_tool": y,
        "prompt_id": prompt_id,
        "text_so_far": np.array([["", ""] for _ in range(n_prompts * k)]),
        "X_7": X.astype(np.float32),
    }


def test_collapse_to_prompts_uses_the_offset_zero_vector_and_the_rate() -> None:
    m = _import()
    p = _payload(signal_in_residual=True)
    xp, rates, ids = m.collapse_to_prompts(p, layer=7, label="is_violation")
    assert xp.shape == (30, 8) and rates.shape == (30,) and len(ids) == 30
    assert np.all((rates >= 0) & (rates <= 1))


def test_loo_ridge_recovers_a_planted_propensity_signal_and_not_noise() -> None:
    m = _import()
    xp, rates, _ = m.collapse_to_prompts(_payload(signal_in_residual=True), layer=7)
    rho_signal = m.loo_spearman(xp, rates)
    xp0, rates0, _ = m.collapse_to_prompts(_payload(signal_in_residual=False), layer=7)
    rho_noise = m.loo_spearman(xp0, rates0)
    assert rho_signal > 0.8
    assert abs(rho_noise) < 0.5


def test_main_reports_residual_vs_text_with_bootstrap_ci(tmp_path: Path) -> None:
    m = _import()
    p = _payload(signal_in_residual=True)
    acts = tmp_path / "acts.npz"
    np.savez(acts, **p)
    prompts = tmp_path / "prompts.json"
    prompts.write_text(json.dumps([f"update order {i} payment" for i in range(30)]))
    out = tmp_path / "rep.json"
    rc = m.main(
        ["--acts", str(acts), "--prompts", str(prompts), "--out", str(out), "--layers", "7"]
    )
    assert rc == 0
    rep = json.loads(out.read_text())
    assert rep["n_prompts"] == 30 and rep["label"] == "is_violation"
    row = rep["layers"][0]
    assert set(row) >= {
        "layer",
        "residual_spearman",
        "residual_ci",
        "text_spearman",
        "text_ci",
        "residual_minus_text_ci",
    }
    assert row["residual_spearman"] > 0.8
    # the planted signal is in the residual, not the (uninformative) prompt text
    assert row["residual_minus_text_ci"][0] > 0.0
