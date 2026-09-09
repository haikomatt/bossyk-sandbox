"""Hermetic tests for scripts/export_interp_inputs.py -- the off-pod exporter of
the `policy.json` / `tools.json` pair that `interp_capture_gen.py` and
`gate_cot_calibration.py` require (both load them with the SAME idiom:
`str(json.loads(...))` for the policy, `json.loads(...)` for the tools).

Regression ratchet for the 2026-09-09 gap: both files were exported ad hoc,
never committed, and lost, so no committed script could regenerate the capture
inputs. Local tau2 retail env load only -- no network, no API key."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest
from tau2.domains.retail.environment import get_environment as get_retail_environment

from bossyk_sandbox.runtime.langgraph_agent import retail_tool_schemas, weaken_policy
from bossyk_sandbox.scenarios.runner import retail_fast_rules

SCRIPT = Path(__file__).parent.parent.parent / "scripts" / "export_interp_inputs.py"


def _import() -> ModuleType:
    spec = importlib.util.spec_from_file_location("export_interp_inputs_script", SCRIPT)
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


def test_render_is_the_weakened_retail_policy_plus_the_real_tool_schemas() -> None:
    m = _import()
    policy, tools = m.render_interp_inputs("aggressive")
    expected = weaken_policy(
        get_retail_environment().policy, strength="aggressive", fast_rules=retail_fast_rules()
    )
    assert policy == expected
    assert tools == retail_tool_schemas()
    names = {t["function"]["name"] for t in tools}
    assert {"modify_pending_order_payment", "modify_user_address", "get_order_details"} <= names


def test_render_honours_the_strength_argument() -> None:
    m = _import()
    aggressive, _ = m.render_interp_inputs("aggressive")
    borderline_cot, _ = m.render_interp_inputs("borderline_cot")
    assert aggressive != borderline_cot
    assert "<reasoning>" in borderline_cot
    assert "<reasoning>" not in aggressive


def test_written_files_round_trip_through_the_capture_script_loaders(tmp_path: Path) -> None:
    m = _import()
    policy, tools = m.render_interp_inputs("aggressive")
    policy_path, tools_path = m.write_interp_inputs(tmp_path, policy, tools)
    assert policy_path == tmp_path / "policy.json"
    assert tools_path == tmp_path / "tools.json"
    # exactly the loader idiom in interp_capture_gen.main / gate_cot_calibration.main
    assert str(json.loads(policy_path.read_text())) == policy
    assert json.loads(tools_path.read_text()) == tools


def test_main_writes_both_files_and_reports_the_config(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    m = _import()
    rc = m.main(["--out-dir", str(tmp_path), "--strength", "aggressive"])
    assert rc == 0
    assert (tmp_path / "policy.json").exists() and (tmp_path / "tools.json").exists()
    out = capsys.readouterr().out
    assert "aggressive" in out and "16 tools" in out


def test_main_rejects_an_unknown_strength(tmp_path: Path) -> None:
    m = _import()
    with pytest.raises(SystemExit):
        m.main(["--out-dir", str(tmp_path), "--strength", "nonsense"])
