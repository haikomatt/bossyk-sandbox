#!/usr/bin/env python
"""Voice-model robustness sweep: runs the live retail H2/H4 bench once per
(agent model x agent config), then tabulates a cross-model comparison.

The thesis: the fast, low-latency models people actually deploy in voice
pipelines are less robust than the frontier model (which is too slow for
voice), so the two-speed gate's value scales with agent weakness. Each model
runs the SAME grounded retail corpus; per model we report structural crossing
rate READ WITH engagement (a 0-reach is only robustness if the model engaged),
gate prevention (harm N->0), and agent inference latency (voice viability).

Agents run on their provider (NVIDIA NIM for the open fast models, Fireworks
for the kimi baseline) via the `AGENT_*` env seam; the deepseek policy judge
stays on Fireworks throughout (Property IX). Each run is a fresh subprocess of
`live_h2h4_bench.py` because that module reads `LIVE_H2_AGENT` at import.

Gated on RUN_VOICE_SWEEP=1 + the agent keys (NVIDIA_API_KEY for NIM,
FIREWORKS_API_KEY for kimi AND the judge). NVIDIA_API_KEY is not in this repo's
.env -- source it (e.g. from the arc-agi-3 .env) before running.

Usage:
    RUN_VOICE_SWEEP=1 NVIDIA_API_KEY=... FIREWORKS_API_KEY=... \\
        uv run python scripts/voice_model_sweep.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bossyk_sandbox.env import load_project_env
from bossyk_sandbox.scoring.model_sweep import build_model_sweep

REPO_ROOT = Path(__file__).parent.parent
BENCH = REPO_ROOT / "scripts" / "live_h2h4_bench.py"
OUTPUT_DIR = REPO_ROOT / "docs" / "bench_output"
CORPUS = REPO_ROOT / "probes" / "grounded" / "retail.json"

NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"


@dataclass(frozen=True)
class SweepModel:
    """One agent model in the sweep, with the provider it runs on. `key_env`
    names the env var holding that provider's key."""

    model: str
    base_url: str
    key_env: str


# The confirmed set (tool-calling smoke-checked). Ordered small -> large so the
# comparison reads along the latency/capability spectrum a voice deployment
# chooses among. llama-3.2-3b was dropped (emits <|python_tag|> text, not
# parsed tool_calls -- footnoted as a can't-tool-call finding); ministral
# dropped (timed out, disqualifying for voice).
MODELS: list[SweepModel] = [
    SweepModel("meta/llama-3.1-8b-instruct", NIM_BASE_URL, "NVIDIA_API_KEY"),
    SweepModel("nvidia/llama-3.3-nemotron-super-49b-v1", NIM_BASE_URL, "NVIDIA_API_KEY"),
    SweepModel("meta/llama-3.3-70b-instruct", NIM_BASE_URL, "NVIDIA_API_KEY"),
    SweepModel("accounts/fireworks/models/kimi-k2p6", FIREWORKS_BASE_URL, "FIREWORKS_API_KEY"),
]
# compliant = each model's own policy-following behavior (inherent robustness);
# weak = the same weakened-policy override across models (shows the gate-save is
# agent-independent -- it catches every model's crossings).
AGENTS: list[str] = ["compliant", "weak"]


def _slug(model: str) -> str:
    return model.split("/")[-1].replace(".", "-")


def build_run_env(
    base_env: Mapping[str, str],
    *,
    model: SweepModel,
    agent: str,
    output_path: Path,
    corpus_path: Path,
) -> dict[str, str]:
    """The env for one bench subprocess: the agent provider via `AGENT_*`, the
    corpus/output/agent-config seams, and the E2E gate. The judge keeps using
    `FIREWORKS_API_KEY` from the inherited env. Raises if the model's provider
    key is missing."""
    key = base_env.get(model.key_env)
    if not key:
        raise RuntimeError(
            f"{model.key_env} is not set -- required to run agent model {model.model!r}"
        )
    return {
        **base_env,
        "RUN_LIVE_H2_E2E": "1",
        "LIVE_H2_DOMAIN": "retail",
        "LIVE_H2_MODE": "single",
        "LIVE_H2_AGENT": agent,
        "LIVE_H2_CORPUS": str(corpus_path),
        "LIVE_H2_OUTPUT": str(output_path),
        "AGENT_MODEL": model.model,
        "AGENT_BASE_URL": model.base_url,
        "AGENT_API_KEY": key,
    }


def run_sweep(
    models: list[SweepModel],
    agents: list[str],
    *,
    base_env: Mapping[str, str],
    output_dir: Path,
    corpus_path: Path,
    run_bench: Callable[[dict[str, str]], None],
    load_result: Callable[[Path], dict[str, Any]],
) -> dict[str, Any]:
    """Run each (model, agent) via `run_bench` (injectable so tests fake the
    subprocess), collect each output via `load_result`, and tabulate the
    cross-model comparison. Rows keep model x agent order."""
    runs: list[tuple[str, str, dict[str, Any]]] = []
    for model in models:
        for agent in agents:
            output_path = output_dir / f"voice_sweep_{agent}_{_slug(model.model)}.json"
            env = build_run_env(
                base_env, model=model, agent=agent, output_path=output_path, corpus_path=corpus_path
            )
            run_bench(env)
            runs.append((model.model, agent, load_result(output_path)))
    return build_model_sweep(runs)


def _run_bench_subprocess(env: dict[str, str]) -> None:
    print(f"  agent={env['AGENT_MODEL']} config={env['LIVE_H2_AGENT']} -> {env['LIVE_H2_OUTPUT']}")
    subprocess.run([sys.executable, str(BENCH)], env=env, check=True)


def main() -> int:
    load_project_env()
    import os

    if os.environ.get("RUN_VOICE_SWEEP") != "1":
        print("Set RUN_VOICE_SWEEP=1 to run the (billable) voice-model sweep.", file=sys.stderr)
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    combined = run_sweep(
        MODELS,
        AGENTS,
        base_env=dict(os.environ),
        output_dir=OUTPUT_DIR,
        corpus_path=CORPUS,
        run_bench=_run_bench_subprocess,
        load_result=lambda path: json.loads(path.read_text()),
    )
    combined["generated_at"] = datetime.now(UTC).isoformat()
    combined["corpus"] = str(CORPUS)
    out = OUTPUT_DIR / "voice_model_sweep_retail.json"
    out.write_text(json.dumps(combined, indent=2))
    print(f"\nWrote cross-model comparison to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
