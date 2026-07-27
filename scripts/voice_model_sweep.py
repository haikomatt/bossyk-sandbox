#!/usr/bin/env python
"""Voice-model robustness sweep: runs the live retail H2/H4 bench once per
(agent model x agent config), then tabulates a cross-model comparison.

The thesis: the fast, low-latency models people actually deploy in voice
pipelines are less robust than the frontier model (which is too slow for
voice), so the two-speed gate's value scales with agent weakness. Each model
runs the SAME grounded retail corpus; per model we report structural crossing
rate READ WITH engagement (a 0-reach is only robustness if the model engaged),
gate prevention (harm N->0), and agent inference latency (voice viability).

Agent models come from the shared registry (`runtime.agent_models`):
self-hosted open-weight models on RunPod serverless (worker-vllm,
OpenAI-compatible) for the fast/open models, Fireworks for the kimi
baseline. A RunPod entry's LIVE endpoint id is runtime state, not a
registry field -- pass it via the `endpoints` map (registry name -> RunPod
endpoint id); a RunPod model with no endpoint id raises loudly rather than
running against a stale/guessed URL. Each provider's model resolves onto
the agent via the `AGENT_*` env seam
(`runtime.langgraph_agent._resolve_agent_config`); the deepseek policy judge
stays on Fireworks throughout (Property IX). Each run is a fresh subprocess
of `live_h2h4_bench.py` because that module reads `LIVE_H2_AGENT` at import.

Gated on RUN_VOICE_SWEEP=1 + the agent keys (RUNPOD_API_KEY for the
self-hosted RunPod models, FIREWORKS_API_KEY for kimi AND the judge) + a
live RunPod endpoint id per RunPod model in the sweep (see
`_endpoint_env_var`).

Usage:
    RUN_VOICE_SWEEP=1 RUNPOD_API_KEY=... FIREWORKS_API_KEY=... \\
        RUNPOD_ENDPOINT_QWEN2_5_7B=... \\
        uv run python scripts/voice_model_sweep.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bossyk_sandbox.env import load_project_env
from bossyk_sandbox.runtime.agent_models import AgentModelSpec, agent_model, openai_base_url_for
from bossyk_sandbox.scoring.model_sweep import build_model_sweep

REPO_ROOT = Path(__file__).parent.parent
BENCH = REPO_ROOT / "scripts" / "live_h2h4_bench.py"
OUTPUT_DIR = REPO_ROOT / "docs" / "bench_output"
CORPUS = REPO_ROOT / "probes" / "grounded" / "retail.json"

# The confirmed set (tool-calling smoke-checked), ordered small -> large so
# the comparison reads along the latency/capability spectrum a voice
# deployment chooses among. Registry names, not raw model strings -- see
# `runtime.agent_models.AGENT_MODELS`.
MODELS: list[str] = ["qwen2.5-7b", "llama-3.1-8b", "kimi"]
# compliant = each model's own policy-following behavior (inherent robustness);
# weak = the same weakened-policy override across models (shows the gate-save is
# agent-independent -- it catches every model's crossings).
AGENTS: list[str] = ["compliant", "weak"]


def _endpoint_env_var(name: str) -> str:
    """The env var a RunPod agent-under-test's LIVE endpoint id is read
    from. Endpoint ids are runtime state (created/destroyed per session by
    `runtime.runpod_serving`), never a registry field -- see
    `runtime.agent_models`'s module docstring."""
    return "RUNPOD_ENDPOINT_" + name.upper().replace("-", "_").replace(".", "_")


def endpoints_from_env(base_env: Mapping[str, str], names: list[str]) -> dict[str, str]:
    """Build the `endpoints` map `run_sweep` needs from whichever
    `RUNPOD_ENDPOINT_*` env vars are set -- only for registry names that are
    actually `runpod_vllm` (Fireworks names need no endpoint id). A missing
    var for a RunPod model is NOT an error here; it surfaces as a loud
    `RuntimeError` from `build_run_env` only if that model is actually run,
    so a partial sweep (e.g. Fireworks-only) doesn't require every RunPod
    endpoint to be up."""
    endpoints: dict[str, str] = {}
    for name in names:
        if agent_model(name).provider != "runpod_vllm":
            continue
        value = base_env.get(_endpoint_env_var(name))
        if value:
            endpoints[name] = value
    return endpoints


def build_run_env(
    base_env: Mapping[str, str],
    *,
    spec: AgentModelSpec,
    agent: str,
    output_path: Path,
    corpus_path: Path,
    endpoints: Mapping[str, str],
) -> dict[str, str]:
    """The env for one bench subprocess: the agent provider via `AGENT_*`
    (resolved from the registry spec), the corpus/output/agent-config seams,
    and the E2E gate. The judge keeps using `FIREWORKS_API_KEY` from the
    inherited env. Raises if the model's provider key is missing, or if a
    RunPod model has no live endpoint id in `endpoints`."""
    key = base_env.get(spec.key_env)
    if not key:
        raise RuntimeError(
            f"{spec.key_env} is not set -- required to run agent model {spec.name!r}"
        )
    if spec.provider == "fireworks":
        if not spec.static_base_url:
            raise RuntimeError(f"agent model {spec.name!r} is fireworks but has no static_base_url")
        base_url = spec.static_base_url
    elif spec.provider == "runpod_vllm":
        endpoint_id = endpoints.get(spec.name)
        if not endpoint_id:
            raise RuntimeError(
                f"no RunPod endpoint id for agent model {spec.name!r} -- pass one via "
                f"endpoints={{{spec.name!r}: <endpoint-id>}} (or set "
                f"{_endpoint_env_var(spec.name)} for the CLI entrypoint); a RunPod model "
                "cannot run against a guessed/stale URL."
            )
        base_url = openai_base_url_for(endpoint_id)
    else:
        raise RuntimeError(f"agent model {spec.name!r} has unknown provider {spec.provider!r}")

    return {
        **base_env,
        "RUN_LIVE_H2_E2E": "1",
        "LIVE_H2_DOMAIN": "retail",
        "LIVE_H2_MODE": "single",
        "LIVE_H2_AGENT": agent,
        "LIVE_H2_CORPUS": str(corpus_path),
        "LIVE_H2_OUTPUT": str(output_path),
        "AGENT_MODEL": spec.model_id,
        "AGENT_BASE_URL": base_url,
        "AGENT_API_KEY": key,
    }


def run_sweep(
    names: list[str],
    agents: list[str],
    *,
    base_env: Mapping[str, str],
    output_dir: Path,
    corpus_path: Path,
    endpoints: Mapping[str, str],
    run_bench: Callable[[dict[str, str]], None],
    load_result: Callable[[Path], dict[str, Any]],
) -> dict[str, Any]:
    """Run each (model, agent) via `run_bench` (injectable so tests fake the
    subprocess), collect each output via `load_result`, and tabulate the
    cross-model comparison. Rows keep model x agent order. `names` are
    registry names (`runtime.agent_models.AGENT_MODELS` keys); the spec's
    `name` (already slug-safe) labels both the output file and the
    comparison row."""
    runs: list[tuple[str, str, dict[str, Any]]] = []
    for name in names:
        spec = agent_model(name)
        for agent in agents:
            output_path = output_dir / f"voice_sweep_{agent}_{spec.name}.json"
            env = build_run_env(
                base_env,
                spec=spec,
                agent=agent,
                output_path=output_path,
                corpus_path=corpus_path,
                endpoints=endpoints,
            )
            run_bench(env)
            runs.append((spec.name, agent, load_result(output_path)))
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

    endpoints = endpoints_from_env(os.environ, MODELS)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    combined = run_sweep(
        MODELS,
        AGENTS,
        base_env=dict(os.environ),
        output_dir=OUTPUT_DIR,
        corpus_path=CORPUS,
        endpoints=endpoints,
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
