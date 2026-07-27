"""The voice-model sweep orchestrator's pure parts: the per-run env it builds
(agent provider resolved from the shared `runtime.agent_models` registry,
judge stays on Fireworks) and the model x agent loop (bench call injected,
so no network / no billable run)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from bossyk_sandbox.runtime.agent_models import agent_model

SCRIPT_PATH = Path(__file__).parents[2] / "scripts" / "voice_model_sweep.py"


def _import_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("voice_model_sweep", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


sweep = _import_script()


# --- registry-driven env resolution ------------------------------------------


def test_build_run_env_resolves_a_runpod_model_via_its_live_endpoint_id() -> None:
    spec = agent_model("qwen2.5-7b")
    base_env = {"RUNPOD_API_KEY": "rp", "FIREWORKS_API_KEY": "fw", "PATH": "/bin"}

    env = sweep.build_run_env(
        base_env,
        spec=spec,
        agent="compliant",
        output_path=Path("/o.json"),
        corpus_path=Path("/c"),
        endpoints={"qwen2.5-7b": "wkdqe0qef23jy2"},
    )

    assert env["AGENT_MODEL"] == "Qwen/Qwen2.5-7B-Instruct"
    assert env["AGENT_BASE_URL"] == "https://api.runpod.ai/v2/wkdqe0qef23jy2/openai/v1"
    assert env["AGENT_API_KEY"] == "rp"  # from the model's key_env
    assert env["FIREWORKS_API_KEY"] == "fw"  # judge key preserved
    assert env["LIVE_H2_AGENT"] == "compliant"
    assert env["LIVE_H2_OUTPUT"] == "/o.json"
    assert env["LIVE_H2_CORPUS"] == "/c"
    assert env["RUN_LIVE_H2_E2E"] == "1"
    assert env["LIVE_H2_DOMAIN"] == "retail"


def test_kimi_baseline_runs_on_its_fireworks_static_base_url() -> None:
    spec = agent_model("kimi")

    env = sweep.build_run_env(
        {"FIREWORKS_API_KEY": "fw"},
        spec=spec,
        agent="weak",
        output_path=Path("/o.json"),
        corpus_path=Path("/c"),
        endpoints={},
    )

    assert env["AGENT_BASE_URL"] == "https://api.fireworks.ai/inference/v1"
    assert env["AGENT_API_KEY"] == "fw"
    assert env["AGENT_MODEL"] == "accounts/fireworks/models/kimi-k2p6"


def test_build_run_env_raises_when_provider_key_missing() -> None:
    spec = agent_model("qwen2.5-7b")

    with pytest.raises(RuntimeError, match="RUNPOD_API_KEY"):
        sweep.build_run_env(
            {"FIREWORKS_API_KEY": "fw"},
            spec=spec,
            agent="compliant",
            output_path=Path("/o"),
            corpus_path=Path("/c"),
            endpoints={"qwen2.5-7b": "ep-1"},
        )


def test_build_run_env_raises_loudly_when_a_runpod_model_has_no_endpoint_id() -> None:
    # A RunPod entry with no live endpoint id must never run against a
    # guessed/stale URL -- raise instead.
    spec = agent_model("qwen2.5-7b")

    with pytest.raises(RuntimeError, match="endpoint"):
        sweep.build_run_env(
            {"RUNPOD_API_KEY": "rp"},
            spec=spec,
            agent="compliant",
            output_path=Path("/o"),
            corpus_path=Path("/c"),
            endpoints={},  # no entry for qwen2.5-7b
        )


# --- endpoint-id env-var seam -------------------------------------------------


def test_endpoint_env_var_uppercases_and_normalizes_the_registry_name() -> None:
    assert sweep._endpoint_env_var("qwen2.5-7b") == "RUNPOD_ENDPOINT_QWEN2_5_7B"
    assert sweep._endpoint_env_var("llama-3.1-8b") == "RUNPOD_ENDPOINT_LLAMA_3_1_8B"


def test_endpoints_from_env_only_picks_up_runpod_models_that_are_set() -> None:
    base_env = {
        "RUNPOD_ENDPOINT_QWEN2_5_7B": "ep-qwen",
        # llama-3.1-8b's var deliberately absent
        "RUNPOD_ENDPOINT_KIMI": "should-be-ignored",  # kimi is fireworks, not runpod
    }

    endpoints = sweep.endpoints_from_env(base_env, ["qwen2.5-7b", "llama-3.1-8b", "kimi"])

    assert endpoints == {"qwen2.5-7b": "ep-qwen"}


# --- the model x agent loop ---------------------------------------------------


def test_run_sweep_runs_each_model_x_agent_and_tabulates(tmp_path: Path) -> None:
    names = ["qwen2.5-7b", "kimi"]
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
        names,
        agents,
        base_env={"RUNPOD_API_KEY": "rp", "FIREWORKS_API_KEY": "fw"},
        output_dir=tmp_path,
        corpus_path=Path("/c"),
        endpoints={"qwen2.5-7b": "ep-1"},
        run_bench=fake_run_bench,
        load_result=fake_load_result,
    )

    # 2 models x 2 agents = 4 runs, in order
    assert len(seen_envs) == 4
    rows = combined["models"]
    assert [(r["model"], r["agent"]) for r in rows] == [
        ("qwen2.5-7b", "compliant"),
        ("qwen2.5-7b", "weak"),
        ("kimi", "compliant"),
        ("kimi", "weak"),
    ]
    # each run got a distinct output path, named from the registry name (no
    # separate _slug() -- spec.name is already slug-safe)
    output_paths = {env["LIVE_H2_OUTPUT"] for env in seen_envs}
    assert len(output_paths) == 4
    assert str(tmp_path / "voice_sweep_compliant_qwen2.5-7b.json") in output_paths
    assert str(tmp_path / "voice_sweep_weak_kimi.json") in output_paths


def test_run_sweep_raises_if_any_included_runpod_model_has_no_endpoint_id() -> None:
    with pytest.raises(RuntimeError, match="endpoint"):
        sweep.run_sweep(
            ["llama-3.1-8b"],
            ["compliant"],
            base_env={"RUNPOD_API_KEY": "rp"},
            output_dir=Path("/out"),
            corpus_path=Path("/c"),
            endpoints={},  # missing
            run_bench=lambda _env: None,
            load_result=lambda _path: {},
        )


def test_no_nim_references_remain_in_the_reconciled_sweep() -> None:
    # The pivot to self-hosted RunPod serving left the old MODELS registry
    # stale (NIM base URLs) -- this is the regression guard that it's gone.
    source = SCRIPT_PATH.read_text()
    assert "NIM" not in source
    assert "nvidia" not in source.lower()
    assert not hasattr(sweep, "SweepModel")
