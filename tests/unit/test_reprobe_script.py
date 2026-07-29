"""Hermetic tests for scripts/reprobe.py -- synthetic activations, no npz IO."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np

SCRIPT_PATH = Path(__file__).parent.parent.parent / "scripts" / "reprobe.py"


def _import_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("reprobe_script", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _payload(n: int = 120, d: int = 40, *, seed: int = 0) -> dict[str, np.ndarray]:
    """Two layers. Policy separable on dim 0; the coherence-error label separable
    on dim 1 (an independent, non-lexical confound); everything else noise."""
    rng = np.random.default_rng(seed)
    is_violation = np.array([i < n // 2 for i in range(n)], dtype=bool)
    is_error = np.where(np.arange(n) % 3 == 0, 1, 0).astype(np.int8)
    x = rng.normal(0.0, 1.0, size=(n, d))
    x[is_violation, 0] += 2.5
    x[~is_violation, 0] -= 2.5
    x[is_error == 1, 1] += 2.5
    return {
        "layers": np.array([0, 7], dtype=np.int64),
        "is_violation": is_violation,
        "is_error": is_error,
        "X_0": x.astype(np.float32),
        "X_7": x.astype(np.float32),
    }


def test_imports_side_effect_free_and_defines_main() -> None:
    module = _import_script()
    assert callable(module.main)
    assert callable(module.reprobe_from_arrays)
    assert callable(module.prepare_labels)


def test_prepare_labels_full_set_when_all_error_known() -> None:
    module = _import_script()
    is_violation = np.array([True, False, True, False])
    is_error = np.array([1, 0, 1, 0], dtype=np.int8)
    mask, labels = module.prepare_labels(is_violation, is_error)
    assert mask.all()
    assert set(labels) == {"policy", "shuffled", "error"}
    assert labels["policy"] == [True, False, True, False]


def test_prepare_labels_restricts_to_labelled_rows_when_partial() -> None:
    module = _import_script()
    is_violation = np.array([True, False, True, False])
    is_error = np.array([1, -1, 0, -1], dtype=np.int8)  # rows 1,3 unjudged
    mask, labels = module.prepare_labels(is_violation, is_error)
    assert mask.tolist() == [True, False, True, False]
    assert labels["policy"] == [True, True]  # only the labelled rows
    assert labels["error"] == [True, False]


def test_prepare_labels_omits_error_when_none_known() -> None:
    module = _import_script()
    is_violation = np.array([True, False, True, False])
    is_error = np.array([-1, -1, -1, -1], dtype=np.int8)
    mask, labels = module.prepare_labels(is_violation, is_error)
    assert mask.all()
    assert "error" not in labels


def test_reprobe_policy_ci_clears_shuffled_and_error_is_independent() -> None:
    module = _import_script()
    report = module.reprobe_from_arrays(_payload(), n_splits=5, n_components=20, n_boot=300, seed=0)
    assert report["n_used"] == 120 and report["has_error_label"] is True
    layer0 = report["layers"]["0"]
    # policy signal is real: its whole CI clears the shuffled floor's CI
    assert layer0["policy"]["ci_lo"] > layer0["shuffled"]["ci_hi"]
    # the error confound is separately decodable (as constructed), and policy
    # does not merely reflect it -- both are strong on their own dims
    assert layer0["error"]["auroc"] > 0.7
