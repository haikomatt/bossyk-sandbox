#!/usr/bin/env python
"""Part A item 1 CLI: parallel, capped, resumable decision generation
(phase-detector-training-step2-datagen.md). Thin CLI over
`bossyk_sandbox.interp.datagen_driver` + `scripts/make_decisions.py`'s
`drive_session` (imported by path, never reimplemented -- this script adds
no new agent-driving logic of its own).

Zero API calls with `--stub`: drives a deterministic scripted LLM that
replays each scenario's OWN authored tool-call sequence (`scenarios/loader.py`
-- both the lookup-then-mutate "benign" scenarios and the mutate-without-lookup
"violation" scenarios are already in the fixture, so replaying them faithfully
produces both classes without guessing). This is what Part A's smoke uses to
prove the harness (caps, concurrency, resume) at the real worker count with
zero spend.

Without `--stub` this drives the REAL weakened live agent (Part B, billable)
and requires `RUN_MAKE_DECISIONS=1`, mirroring `make_decisions.py`'s own gate.
Part B is NOT run by this handoff -- do not set `RUN_MAKE_DECISIONS=1` without
Matt's explicit spend authorization.

Usage (Part A, hermetic, zero cost):
    uv run python scripts/generate_corpus.py --domain retail --stub \\
        --variants-per-scenario 2 --workers 12 --max-usd 20

Usage (Part B, billable, GATED -- do not run without authorization):
    RUN_MAKE_DECISIONS=1 uv run python scripts/generate_corpus.py \\
        --domain retail --variants-per-scenario 400 --workers 13 \\
        --max-usd 20 --cost-per-call-usd <Matt-confirmed Part-B rate>
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

from langchain_core.messages import AIMessage

from bossyk_sandbox.domains import domain_config
from bossyk_sandbox.interp.datagen_driver import (
    CapConfig,
    GenerationSummary,
    GenTask,
    build_scenario_tasks,
    generate_corpus,
)
from bossyk_sandbox.runtime.langgraph_agent import (
    build_weakened_advice_eligibility_agent_session,
    build_weakened_airline_agent_session,
    build_weakened_retail_agent_session,
)
from bossyk_sandbox.scenarios.loader import Scenario, load_scenarios

_BUILDERS: dict[str, Any] = {
    "retail": build_weakened_retail_agent_session,
    "airline": build_weakened_airline_agent_session,
    "advice-eligibility": build_weakened_advice_eligibility_agent_session,
}

DATA_ROOT = Path(__file__).parent.parent / "probes" / "detector" / "data"

_MAKE_DECISIONS_PATH = Path(__file__).parent / "make_decisions.py"


def _load_make_decisions() -> ModuleType:
    """Imports make_decisions.py by path (it's a script, not a package
    module) so this CLI reuses its `drive_session` -- the one place agent
    turns actually get driven -- rather than reimplementing it."""
    spec = importlib.util.spec_from_file_location(
        "make_decisions_for_generate_corpus", _MAKE_DECISIONS_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass
class StubScenarioLLM:
    """Deterministic, zero-network scripted LLM: replays a scenario's OWN
    authored `steps` tool-call sequence in order (both "benign"
    lookup-then-mutate and "violation" mutate-without-lookup scenarios are
    already in the fixture, per `scenarios/loader.py`), then answers with
    plain text once the script is exhausted. Faithful to the scenario's
    intended boundary_label without this script having to guess it."""

    steps: list[Any]
    calls: int = 0
    seen: list[Any] = field(default_factory=list)

    def invoke(self, messages: Any) -> AIMessage:
        self.seen.append(messages)
        if self.calls < len(self.steps):
            step = self.steps[self.calls]
            response = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": step.proposed.tool_name,
                        "args": dict(step.proposed.arguments),
                        "id": f"stub-call-{self.calls}",
                        "type": "tool_call",
                    }
                ],
            )
        else:
            response = AIMessage(content="done")
        self.calls += 1
        return response


def stub_llm_factory_for_domain(domain: str) -> Any:
    """Builds a `llm_factory(task) -> StubScenarioLLM` closure that looks up
    `task.scenario_id`'s real authored step sequence from the domain's own
    scenario fixture (the same one `build_scenario_tasks` reads task
    metadata from) and replays it. A fresh `StubScenarioLLM` per call, so
    concurrent tasks never share the `.calls` counter."""
    cfg = domain_config(domain)
    scenarios: dict[str, Scenario] = {s.scenario_id: s for s in load_scenarios(cfg.scenarios_path)}

    def factory(task: GenTask) -> StubScenarioLLM:
        scenario = scenarios[task.scenario_id]
        return StubScenarioLLM(steps=list(scenario.steps))

    return factory


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--domain", choices=sorted(_BUILDERS), default="retail")
    parser.add_argument(
        "--variants-per-scenario",
        type=int,
        default=1,
        help="independent repeated draws per real scenario, for volume",
    )
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--max-usd", type=float, default=20.0)
    parser.add_argument("--max-calls", type=int, default=None)
    parser.add_argument("--max-wall-min", type=float, default=None)
    parser.add_argument(
        "--cost-per-call-usd",
        type=float,
        default=0.002,
        help="placeholder estimate pending Part B's confirmed provider/model rate",
    )
    parser.add_argument("--task-timeout-sec", type=float, default=180.0)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="defaults to probes/detector/data/<domain>/decisions.jsonl",
    )
    parser.add_argument(
        "--stub",
        action="store_true",
        help="zero-cost: replay each scenario's own authored tool-call script, not a live agent",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if not args.stub and os.environ.get("RUN_MAKE_DECISIONS") != "1":
        print(
            "Set RUN_MAKE_DECISIONS=1 for a live (billable) run, or pass --stub for a "
            "zero-cost dry run.",
            file=sys.stderr,
        )
        return 1

    checkpoint_path = args.checkpoint or (DATA_ROOT / args.domain / "decisions.jsonl")
    tasks = build_scenario_tasks(args.domain, variants_per_scenario=args.variants_per_scenario)
    make_decisions = _load_make_decisions()  # drive_session is reused either way

    if args.stub:
        llm_factory = stub_llm_factory_for_domain(args.domain)
        generator_label = "stub"
    else:

        def llm_factory(_task: GenTask) -> None:
            return None  # let the domain builder resolve a real live LLM

        generator_label = "live"

    summary: GenerationSummary = generate_corpus(
        tasks,
        build_session_fn=_BUILDERS[args.domain],
        llm_factory=llm_factory,
        drive_session_fn=make_decisions.drive_session,
        checkpoint_path=checkpoint_path,
        caps=CapConfig(
            max_usd=args.max_usd,
            max_calls=args.max_calls,
            max_wall_min=args.max_wall_min,
            cost_per_call_usd=args.cost_per_call_usd,
        ),
        workers=args.workers,
        task_timeout_sec=args.task_timeout_sec,
        generator_label=generator_label,
    )

    print(
        f"domain={summary.domain} total={summary.total_tasks} already_done={summary.already_done} "
        f"completed={summary.completed} failed={summary.failed} timed_out={summary.timed_out} "
        f"calls={summary.calls_made} est_usd={summary.estimated_usd:.4f} "
        f"elapsed_min={summary.elapsed_min:.2f} cap_hit={summary.cap_hit} "
        f"decisions={summary.decisions_written} violations={summary.violations_written}"
    )
    print(f"wrote {checkpoint_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
