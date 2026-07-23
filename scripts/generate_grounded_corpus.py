#!/usr/bin/env python
"""Generate an agent-grounded adversarial corpus for the live H2/H4 run
(phase-agent-grounded-live-h2h4.md, path A). Runs the grounded red-team
adversary (conditions.fireworks_adversary, given the target agent's REAL
tau2 tool schemas via tool_context) over the domain's attack-class x
consequence-boundary grid, and freezes every usable attack into
`probes/grounded/<domain>.json` -- the corpus the live bench then replays
(point scripts/live_h2h4_bench.py at it with LIVE_H2_CORPUS).

Fixes the L1 root cause (docs/phase-live-h2h4-L1-results.md): the H1
adversary was never given the agent's tools, so its payloads referenced
tools the retail agent lacks and 0/25 reached a boundary. The grounded
adversary attacks the agent's actual toolset.

Domain via GROUNDED_DOMAIN (default retail -- retail-primary), per-cell
attack budget via GROUNDED_BUDGET (default 2 -- the cheap A-first probe),
adversary via GROUNDED_ADVERSARY (default fireworks-deepseek -- the only
viable H1 adversary; Fable refuses 100%, plan section 15A). Property IX: the
live AGENT is kimi-k2p6; the adversary here only writes attack text.

Gated on FIREWORKS_API_KEY + RUN_GROUNDED_GEN=1 -- real, billable adversary
API traffic, never run in the deterministic suite (a smoke test imports this
module and checks it defines main; it does NOT call main()).

Usage:
    RUN_GROUNDED_GEN=1 FIREWORKS_API_KEY=... \\
        uv run python scripts/generate_grounded_corpus.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from bossyk_sandbox.conditions.adversary_registry import build_adversary
from bossyk_sandbox.conditions.fireworks_adversary import render_tool_context
from bossyk_sandbox.conditions.grid import AttackClass, ProbeCell, boundaries_for, build_grid
from bossyk_sandbox.conditions.grounded_corpus import generate_grounded_attempts
from bossyk_sandbox.conditions.live_boundary import structural_boundaries
from bossyk_sandbox.conditions.retention import freeze_attempt, save_regression_probes
from bossyk_sandbox.env import load_project_env
from bossyk_sandbox.runtime.langgraph_agent import retail_tool_schemas
from bossyk_sandbox.scoring.cost import build_token_ledger

REPO_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = REPO_ROOT / "probes" / "grounded"

DOMAIN = os.environ.get("GROUNDED_DOMAIN", "retail")
BUDGET = int(os.environ.get("GROUNDED_BUDGET", "2"))
ADVERSARY = os.environ.get("GROUNDED_ADVERSARY", "fireworks-deepseek")
# "single" (path A: full grid, single-turn) | "goal" (path B: structural, multi-turn
# goals) | "structural" (dir 1: structural boundaries, single-turn, oracle-tool-pinned)
GROUNDED_MODE = os.environ.get("GROUNDED_MODE", "single")

# Per-domain source of the REAL tool schemas the live agent binds -- the
# grounding the adversary attacks. Retail-primary; airline can be added when
# a live airline grounded run needs it.
_TOOL_SCHEMAS_BY_DOMAIN = {
    "retail": retail_tool_schemas,
}


def _output_path(domain: str) -> Path:
    suffix = {"goal": "-multiturn", "structural": "-structural"}.get(GROUNDED_MODE, "")
    return OUTPUT_DIR / f"{domain}{suffix}.json"


def _cells(domain: str) -> list[ProbeCell]:
    # goal (path B) + structural (dir 1) target only oracle-scorable structural
    # boundaries, one representative class (tool_misuse) per boundary. single
    # (path A) uses the full attack-class x boundary grid.
    if GROUNDED_MODE in {"goal", "structural"}:
        return build_grid(domain, [AttackClass.TOOL_MISUSE], structural_boundaries(domain))
    return build_grid(domain, list(AttackClass), boundaries_for(domain))


def _real_mode_requested() -> bool:
    if os.environ.get("RUN_GROUNDED_GEN") != "1":
        print("Set RUN_GROUNDED_GEN=1 to generate the grounded corpus.", file=sys.stderr)
        raise SystemExit(1)
    if not os.environ.get("FIREWORKS_API_KEY"):
        print("FIREWORKS_API_KEY is required to generate the grounded corpus.", file=sys.stderr)
        raise SystemExit(1)
    if DOMAIN not in _TOOL_SCHEMAS_BY_DOMAIN:
        print(
            f"GROUNDED_DOMAIN={DOMAIN!r} has no tool-schema source "
            f"(registered: {sorted(_TOOL_SCHEMAS_BY_DOMAIN)}).",
            file=sys.stderr,
        )
        raise SystemExit(1)
    if GROUNDED_MODE not in {"single", "goal", "structural"}:
        print(
            f"GROUNDED_MODE={GROUNDED_MODE!r} must be 'single', 'goal', or 'structural'.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    return True


def main() -> None:
    load_project_env()
    _real_mode_requested()

    tool_context = render_tool_context(_TOOL_SCHEMAS_BY_DOMAIN[DOMAIN]())
    adversary = build_adversary(
        ADVERSARY, tool_context=tool_context, goal_mode=(GROUNDED_MODE == "goal")
    )
    cells = _cells(DOMAIN)

    print(f"=== Grounded corpus generation (domain={DOMAIN}) ===")
    print(f"adversary: {ADVERSARY}")
    print(f"mode: {GROUNDED_MODE}")
    print(f"grid: {len(cells)} cells x budget {BUDGET} = {len(cells) * BUDGET} attempts")
    print(f"grounded in {len(tool_context.splitlines())} real tools\n")

    attempts = generate_grounded_attempts(adversary, cells, BUDGET)
    probes = [freeze_attempt(a) for a in attempts if a.status == "ok"]

    n_ok = sum(1 for a in attempts if a.status == "ok")
    n_refused = sum(1 for a in attempts if a.status == "refused")
    n_error = sum(1 for a in attempts if a.status == "error")
    print(f"attempts: {len(attempts)} (ok={n_ok} refused={n_refused} error={n_error})")
    print(f"froze {len(probes)} grounded probes.\n")

    ledger = build_token_ledger(attempts)
    print("-- adversary token ledger --")
    for model, entry in ledger.items():
        print(
            f"  {model}: {entry.calls} calls ({entry.refused_calls} refused), "
            f"total_tokens={entry.usage.total_tokens}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = _output_path(DOMAIN)
    save_regression_probes(probes, output_path)
    print(f"\nWrote {len(probes)} probes to {output_path}")


if __name__ == "__main__":
    main()
