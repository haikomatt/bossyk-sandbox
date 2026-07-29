"""Hermetic tests for scripts/interp_capture.py.

Imports the script by path (side-effect-free) and exercises the pure helpers.
Never calls main(), never loads a model, never imports torch/nnsight.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT_PATH = Path(__file__).parent.parent.parent / "scripts" / "interp_capture.py"


def _import_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("interp_capture_script", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_imports_side_effect_free_and_defines_main() -> None:
    module = _import_script()
    assert callable(module.main)
    assert callable(module.nnsight_tracer)  # defined, but not called (no torch here)


def test_load_items_parses_optional_error_label() -> None:
    module = _import_script()
    items = module.load_items(
        [
            {"step_id": "s1", "prompt": "p1", "is_violation": True, "is_error": False},
            {"step_id": "s2", "prompt": "p2", "is_violation": False},  # is_error absent -> None
        ]
    )
    assert len(items) == 2
    assert items[0].is_violation is True and items[0].is_error is False
    assert items[1].is_error is None


def test_parse_layers() -> None:
    module = _import_script()
    assert module.parse_layers("0,8,16,24,31") == [0, 8, 16, 24, 31]
    assert module.parse_layers("4") == [4]


def test_parse_layers_rejects_empty() -> None:
    module = _import_script()
    with pytest.raises(ValueError, match="no layers"):
        module.parse_layers(" , ")
