"""Hermetic tests for scripts/interp_pod.py.

Imports the script by path (it is side-effect-free at import) and exercises
`bring_up` / `deadline_iso` with an injected `run` and a fixed `now`. Never calls
`main()`, never touches the network -- mirrors tests/unit/test_smoke_tool_call.py.
"""

from __future__ import annotations

import importlib.util
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT_PATH = Path(__file__).parent.parent.parent / "scripts" / "interp_pod.py"
NOW = datetime(2026, 7, 28, 12, 0, 0, tzinfo=UTC)


def _import_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("interp_pod_script", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_script_imports_side_effect_free_and_defines_main() -> None:
    module = _import_script()
    assert callable(module.main)
    assert callable(module.bring_up)


def test_deadline_iso_is_now_plus_hours_utc_z() -> None:
    module = _import_script()
    assert module.deadline_iso(NOW, 6.0) == "2026-07-28T18:00:00Z"


def test_bring_up_gated_model_requires_hf_token() -> None:
    module = _import_script()
    calls: list[list[str]] = []
    with pytest.raises(RuntimeError, match="gated"):
        module.bring_up(
            model_name="llama-3.1-8b",
            gpu_id="NVIDIA A40",
            image="img",
            container_disk_gb=80,
            hours=6.0,
            now=NOW,
            env={},  # no HF_TOKEN
            run=calls.append,
        )
    assert calls == []  # nothing reached the injected run


def test_bring_up_happy_path_creates_with_computed_deadline() -> None:
    module = _import_script()
    calls: list[list[str]] = []
    result, deadline = module.bring_up(
        model_name="llama-3.1-8b",
        gpu_id="NVIDIA A40",
        image="runpod/pytorch:cuda",
        container_disk_gb=80,
        hours=6.0,
        now=NOW,
        env={"HF_TOKEN": "unused-in-argv"},
        run=calls.append,
    )
    assert deadline == "2026-07-28T18:00:00Z"
    assert len(calls) == 1
    argv = calls[0]
    assert argv[:3] == ["runpodctl", "pod", "create"]
    assert argv[argv.index("--name") + 1] == "interp-llama-3.1-8b"
    assert argv[argv.index("--terminate-after") + 1] == "2026-07-28T18:00:00Z"
    # the real token never appears in argv (only the literal placeholder)
    assert not any(part == "unused-in-argv" for part in argv)


def test_bring_up_ungated_model_needs_no_token() -> None:
    module = _import_script()
    calls: list[list[str]] = []
    module.bring_up(
        model_name="qwen2.5-7b",
        gpu_id="NVIDIA A40",
        image="img",
        container_disk_gb=80,
        hours=3.0,
        now=NOW,
        env={},
        run=calls.append,
    )
    assert calls[0].index("--terminate-after") >= 0
