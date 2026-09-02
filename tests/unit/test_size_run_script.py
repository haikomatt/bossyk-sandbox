"""Hermetic tests for scripts/size_run.py (hermetic diversity fix, item 4).
Pure arithmetic CLI over `bossyk_sandbox.interp.size_calc` -- no network, no
LLM, no filesystem writes.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT_PATH = Path(__file__).parent.parent.parent / "scripts" / "size_run.py"


def _import_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("size_run_cli_test", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_imports_side_effect_free_and_defines_main() -> None:
    module = _import_script()
    assert callable(module.main)


def test_arg_parser_defaults_targets_to_150_and_300() -> None:
    module = _import_script()
    args = module._build_arg_parser().parse_args(
        [
            "--domain",
            "retail",
            "--unique-rate",
            "0.5",
            "--violation-rate",
            "0.1",
            "--cost-per-call-usd",
            "0.002",
        ]
    )
    assert args.targets == [150, 300]


def test_main_prints_one_line_per_target(capsys: pytest.CaptureFixture[str]) -> None:
    module = _import_script()
    rc = module.main(
        [
            "--domain",
            "retail",
            "--unique-rate",
            "0.5",
            "--violation-rate",
            "0.1",
            "--cost-per-call-usd",
            "0.002",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    lines = [line for line in out.splitlines() if line.strip()]
    assert len(lines) == 2  # one per default target (150, 300)
    assert "target=150" in lines[0]
    assert "target=300" in lines[1]
    # 150/0.1=1500 unique -> /0.5=3000 calls -> $6.00
    assert "calls=3000" in lines[0]
    assert "6.00" in lines[0]


def test_main_accepts_custom_targets(capsys: pytest.CaptureFixture[str]) -> None:
    module = _import_script()
    rc = module.main(
        [
            "--domain",
            "retail",
            "--unique-rate",
            "0.5",
            "--violation-rate",
            "0.1",
            "--cost-per-call-usd",
            "0.002",
            "--targets",
            "200",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    lines = [line for line in out.splitlines() if line.strip()]
    assert len(lines) == 1
    assert "target=200" in lines[0]


def test_main_rejects_invalid_rates_without_a_traceback(
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _import_script()
    rc = module.main(
        [
            "--domain",
            "retail",
            "--unique-rate",
            "0.0",  # invalid: size_calc.size_run requires (0, 1]
            "--violation-rate",
            "0.1",
            "--cost-per-call-usd",
            "0.002",
        ]
    )
    assert rc == 1
