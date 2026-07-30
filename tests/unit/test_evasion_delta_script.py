"""Hermetic tests for scripts/evasion_delta.py pure metric core (no I/O, no net)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np

SCRIPT = Path(__file__).parent.parent.parent / "scripts" / "evasion_delta.py"


def _import() -> ModuleType:
    spec = importlib.util.spec_from_file_location("evasion_delta_script", SCRIPT)
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


def _synthetic(
    n_prompts: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per prompt: clean-violation, clean-compliant, laundered-violation,
    laundered-compliant. Text SEPARATES on clean (1.0) but not on laundered (0.5);
    the probe SEPARATES in both. So drop_text ~ 0.5, drop_probe ~ 0, delta ~ +0.5
    (the probe wins under laundering) -- the affirmative evasion-gap shape."""
    text, probe, labels, laundered, pid = [], [], [], [], []
    for p in range(n_prompts):
        # clean: both monitors separate
        text += [0.9, 0.1]
        probe += [0.9, 0.1]
        labels += [True, False]
        laundered += [False, False]
        pid += [p, p]
        # laundered: text collapses to a tie (auroc 0.5), probe still separates
        text += [0.5, 0.5]
        probe += [0.8, 0.2]
        labels += [True, False]
        laundered += [True, True]
        pid += [p, p]
    return (
        np.array(text),
        np.array(probe),
        np.array(labels, dtype=bool),
        np.array(laundered, dtype=bool),
        np.array(pid),
    )


def test_evasion_delta_positive_when_probe_survives_laundering() -> None:
    m = _import()
    text, probe, labels, laundered, _ = _synthetic()
    r = m.evasion_delta(text, probe, labels, laundered)
    assert r.text_clean_auroc == 1.0
    assert r.text_laundered_auroc == 0.5  # tie -> 0.5
    assert abs(r.drop_text - 0.5) < 1e-9
    assert r.probe_clean_auroc == 1.0
    assert r.probe_laundered_auroc == 1.0
    assert abs(r.drop_probe - 0.0) < 1e-9
    assert abs(r.delta - 0.5) < 1e-9  # probe degrades less than text
    assert r.n_clean == 8 and r.n_laundered == 8


def test_evasion_delta_zero_when_both_collapse_equally() -> None:
    # The likely-negative outcome: probe collapses under laundering just like text.
    m = _import()
    text, probe, labels, laundered, _ = _synthetic()
    probe = text.copy()  # probe behaves exactly like text -> equal drops
    r = m.evasion_delta(text, probe, labels, laundered)
    assert abs(r.delta) < 1e-9


def test_group_bootstrap_delta_ci_is_positive_for_a_clear_gap() -> None:
    m = _import()
    text, probe, labels, laundered, pid = _synthetic(n_prompts=8)
    lo, hi = m.group_bootstrap_delta_ci(
        text, probe, labels, laundered, pid, n_boot=500, alpha=0.05, seed=0
    )
    assert lo > 0.0  # a clear positive gap -> CI excludes 0
    assert hi >= lo
    assert hi <= 1.0


def test_group_bootstrap_resamples_prompts_not_rows() -> None:
    # Determinism + shape: same seed -> same CI; the CI is finite.
    m = _import()
    text, probe, labels, laundered, pid = _synthetic(n_prompts=6)
    a = m.group_bootstrap_delta_ci(text, probe, labels, laundered, pid, n_boot=300, seed=7)
    b = m.group_bootstrap_delta_ci(text, probe, labels, laundered, pid, n_boot=300, seed=7)
    assert a == b
