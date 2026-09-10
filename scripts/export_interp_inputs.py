#!/usr/bin/env python
"""Export the off-pod inputs the interp capture scripts require: `policy.json`
(the weakened retail system policy, as a JSON string) and `tools.json` (the 16
real tau2 retail tool schemas, as a JSON list).

`scripts/interp_capture_gen.py` and `scripts/gate_cot_calibration.py` both take
`--policy-file` / `--tools-file` because the capture pod has only
nnsight/transformers/numpy and must never import the bossyk agent stack. Until
2026-09-09 nothing in the repo produced those two files: they were exported ad
hoc and lost with the interp worktree, which is why a previous run's activations
could not be re-labelled. This script is the committed producer.

`--strength` selects the weakening override (`weaken_policy`): `aggressive` is
the immediate-action H1 regime that produced `leadtime_report.json` (26
borderline prompts x 12 rollouts, temp 1.0 -- see
probes/interp/results/RESULTS.md); `borderline_cot` is the reasoning-window
regime `gate_cot_calibration.py` gates. Local tau2 env load only, no network.

Usage (tau2 reads its data tree from TAU2_DATA_DIR, as tests/conftest.py sets):
    TAU2_DATA_DIR=../tau2-bench/data uv run python scripts/export_interp_inputs.py \\
        --out-dir .interp_data --strength aggressive
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from tau2.domains.retail.environment import get_environment as get_retail_environment

from bossyk_sandbox.runtime.langgraph_agent import (
    _WEAKENING_OVERRIDE_TEMPLATES,
    retail_tool_schemas,
    weaken_policy,
)
from bossyk_sandbox.scenarios.runner import retail_fast_rules

STRENGTHS = sorted(_WEAKENING_OVERRIDE_TEMPLATES)


def render_interp_inputs(strength: str) -> tuple[str, list[dict[str, Any]]]:
    """The (weakened retail policy, retail tool schemas) pair exactly as the
    weakened retail agent session renders them: the real tau2 retail policy
    plus the requested override, with tool names read off `retail_fast_rules()`
    so the prompt and the gate agree on which tools are gated."""
    policy = weaken_policy(
        get_retail_environment().policy, strength=strength, fast_rules=retail_fast_rules()
    )
    return policy, retail_tool_schemas()


def write_interp_inputs(
    out_dir: Path, policy: str, tools: list[dict[str, Any]]
) -> tuple[Path, Path]:
    """Write `policy.json` (a JSON string) and `tools.json` (a JSON list) under
    `out_dir`, in the exact shapes the capture scripts' loaders expect."""
    out_dir.mkdir(parents=True, exist_ok=True)
    policy_path = out_dir / "policy.json"
    tools_path = out_dir / "tools.json"
    policy_path.write_text(json.dumps(policy))
    tools_path.write_text(json.dumps(tools, indent=2))
    return policy_path, tools_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="export policy.json + tools.json for interp capture"
    )
    parser.add_argument("--out-dir", required=True, help="directory to write the two files into")
    parser.add_argument(
        "--strength",
        choices=STRENGTHS,
        default="aggressive",
        help="weakening override (default: aggressive = the immediate-action H1 regime)",
    )
    args = parser.parse_args(argv)

    policy, tools = render_interp_inputs(args.strength)
    policy_path, tools_path = write_interp_inputs(Path(args.out_dir), policy, tools)
    print(
        f"strength={args.strength} policy={len(policy)} chars -> {policy_path}; "
        f"{len(tools)} tools -> {tools_path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
