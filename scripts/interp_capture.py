#!/usr/bin/env python
"""On-pod activation capture: nnsight tracer + gated CLI (Phase 4).

Runs ON the interp pod (where torch + nnsight are installed). Reads a JSON of
labelled decision items, reads the agent model's residual stream at the requested
layers (last context token = T4), and writes a per-layer probe report
(policy AUROC vs the shuffled floor vs the general-failure confound) via
`interp.capture_run`.

The heavy work is isolated: `nnsight_tracer` is the only thing that imports
torch/nnsight, and it does so lazily (inside the factory) so this module imports
side-effect-free in the default test suite -- the tests exercise the pure helpers
(`load_items`, `parse_layers`) by path and never load a model.

Gated on RUN_INTERP_CAPTURE=1: loading an 8B model + tracing is GPU work, only
sensible on the pod.

Items JSON: a list of {step_id, prompt, is_violation, is_error?}. Produced from a
behavioral run's captured decision prompts (a small follow-up producer).

Usage (on the pod):
    RUN_INTERP_CAPTURE=1 uv run python scripts/interp_capture.py \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --items decisions.json --layers 0,8,16,24,31 --out capture_report.json

NOTE: the residual-access path (`model.model.layers[L].output[0]`) is Llama-
shaped and nnsight-version-sensitive -- confirm it on the pod on first run; it is
deliberately the single spot to adjust if an nnsight upgrade moves the proxy API.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from bossyk_sandbox.env import load_project_env
from bossyk_sandbox.interp.capture_run import DecisionItem, Tracer, run_capture_report


def load_items(data: list[dict[str, Any]]) -> list[DecisionItem]:
    """Parse the items JSON into DecisionItems. `is_error` is optional (absent ->
    None, so the confound probe is skipped unless every item supplies it)."""
    return [
        DecisionItem(
            step_id=str(row["step_id"]),
            prompt=str(row["prompt"]),
            is_violation=bool(row["is_violation"]),
            is_error=None if row.get("is_error") is None else bool(row["is_error"]),
        )
        for row in data
    ]


def parse_layers(spec: str) -> list[int]:
    """Parse a comma-separated layer list, e.g. '0,8,16,24,31'."""
    layers = [int(part) for part in spec.split(",") if part.strip()]
    if not layers:
        raise ValueError("no layers parsed from --layers")
    return layers


def nnsight_tracer(model_id: str, *, device_map: str = "auto") -> Tracer:
    """Build an nnsight-backed tracer: load the model once, return a closure that
    caches the residual stream at the LAST context token for the requested layers.
    Imports torch/nnsight lazily (pod-only). Returns plain float lists so the
    orchestration stays torch-free downstream."""
    from nnsight import LanguageModel  # noqa: PLC0415 -- lazy, pod-only heavy import

    model = LanguageModel(model_id, device_map=device_map)

    def tracer(prompt: str, layers: list[int]) -> dict[int, list[float]]:
        with model.trace(prompt):
            saved = {
                layer: model.model.layers[layer].output[0][0, -1, :].save() for layer in layers
            }
        return {layer: proxy.value.float().cpu().tolist() for layer, proxy in saved.items()}

    return tracer


def main(argv: list[str] | None = None) -> int:
    load_project_env()
    parser = argparse.ArgumentParser(description="on-pod activation capture + probe")
    parser.add_argument("--model", required=True, help="HF model id (the agent under test)")
    parser.add_argument("--items", required=True, help="path to the decisions JSON")
    parser.add_argument("--layers", required=True, help="comma-separated layers, e.g. 0,8,16,24,31")
    parser.add_argument("--out", required=True, help="path to write the probe report JSON")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    if os.environ.get("RUN_INTERP_CAPTURE") != "1":
        print("Set RUN_INTERP_CAPTURE=1 to run the (GPU) capture on the pod.", file=sys.stderr)
        return 1

    items = load_items(json.loads(Path(args.items).read_text()))
    layers = parse_layers(args.layers)
    tracer = nnsight_tracer(args.model)
    report = run_capture_report(items, layers, tracer, seed=args.seed)

    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"wrote {args.out}: {report['n_violation']}/{report['n_items']} violation items")
    for layer, scores in report["layers"].items():
        print(f"  layer {layer}: {scores}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
