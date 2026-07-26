"""The voice-model sweep orchestrator's pure parts: the per-run env it builds
(agent provider via AGENT_*, judge stays on Fireworks) and the model x agent
loop (bench call injected, so no network / no billable run)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

SCRIPT_PATH = Path(__file__).parents[2] / "scripts" / "voice_model_sweep.py"


def _import_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("voice_model_sweep", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


sweep = _import_script()


def test_build_run_env_points_the_agent_at_its_provider_and_gates_e2e() -> None:
    model = sweep.SweepModel("meta/llama-3.1-8b-instruct", sweep.NIM_BASE_URL, "NVIDIA_API_KEY")
    base_env = {"NVIDIA_API_KEY": "nv", "FIREWORKS_API_KEY": "fw", "PATH": "/bin"}

    env = sweep.build_run_env(
        base_env,
        model=model,
        agent="compliant",
        output_path=Path("/o.json"),
        corpus_path=Path("/c"),
    )

    assert env["AGENT_MODEL"] == "meta/llama-3.1-8b-instruct"
    assert env["AGENT_BASE_URL"] == sweep.NIM_BASE_URL
    assert env["AGENT_API_KEY"] == "nv"  # from the model's key_env
    assert env["FIREWORKS_API_KEY"] == "fw"  # judge key preserved
    assert env["LIVE_H2_AGENT"] == "compliant"
    assert env["LIVE_H2_OUTPUT"] == "/o.json"
    assert env["LIVE_H2_CORPUS"] == "/c"
    assert env["RUN_LIVE_H2_E2E"] == "1"
    assert env["LIVE_H2_DOMAIN"] == "retail"


def test_kimi_baseline_runs_on_fireworks() -> None:
    model = sweep.SweepModel(
        "accounts/fireworks/models/kimi-k2p6", sweep.FIREWORKS_BASE_URL, "FIREWORKS_API_KEY"
    )
    env = sweep.build_run_env(
        {"FIREWORKS_API_KEY": "fw"},
        model=model,
        agent="weak",
        output_path=Path("/o.json"),
        corpus_path=Path("/c"),
    )

    assert env["AGENT_BASE_URL"] == sweep.FIREWORKS_BASE_URL
    assert env["AGENT_API_KEY"] == "fw"


def test_build_run_env_raises_when_provider_key_missing() -> None:
    model = sweep.SweepModel("meta/llama-3.1-8b-instruct", sweep.NIM_BASE_URL, "NVIDIA_API_KEY")

    with pytest.raises(RuntimeError, match="NVIDIA_API_KEY"):
        sweep.build_run_env(
            {"FIREWORKS_API_KEY": "fw"},
            model=model,
            agent="compliant",
            output_path=Path("/o"),
            corpus_path=Path("/c"),
        )


def test_run_sweep_runs_each_model_x_agent_and_tabulates(tmp_path: Path) -> None:
    models = [
        sweep.SweepModel("meta/llama-3.1-8b-instruct", sweep.NIM_BASE_URL, "NVIDIA_API_KEY"),
        sweep.SweepModel(
            "accounts/fireworks/models/kimi-k2p6", sweep.FIREWORKS_BASE_URL, "FIREWORKS_API_KEY"
        ),
    ]
    agents = ["compliant", "weak"]
    seen_envs: list[dict[str, str]] = []

    def fake_run_bench(env: dict[str, str]) -> None:
        seen_envs.append(env)

    def fake_load_result(_path: Path) -> dict[str, Any]:
        return {
            "crossings": [{"reached": False}],
            "engagement": {"n": 1, "successes": 1, "rate": 1.0},
            "live_h4": {"n_violations": 0, "prevented": 0, "harm_delta": 0},
            "latency": {},
        }

    combined = sweep.run_sweep(
        models,
        agents,
        base_env={"NVIDIA_API_KEY": "nv", "FIREWORKS_API_KEY": "fw"},
        output_dir=tmp_path,
        corpus_path=Path("/c"),
        run_bench=fake_run_bench,
        load_result=fake_load_result,
    )

    # 2 models x 2 agents = 4 runs, in order
    assert len(seen_envs) == 4
    rows = combined["models"]
    assert [(r["model"], r["agent"]) for r in rows] == [
        ("meta/llama-3.1-8b-instruct", "compliant"),
        ("meta/llama-3.1-8b-instruct", "weak"),
        ("accounts/fireworks/models/kimi-k2p6", "compliant"),
        ("accounts/fireworks/models/kimi-k2p6", "weak"),
    ]
    # each run got a distinct output path
    assert len({env["LIVE_H2_OUTPUT"] for env in seen_envs}) == 4
