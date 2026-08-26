"""Parallel, capped, resumable decision-generation driver
(phase-detector-training-step2-datagen.md Part A, item 1).

THIN WRAPPER, not a parallel reimplementation: the actual agent-driving logic
stays exactly `scripts/make_decisions.py::drive_session` (imported by the
caller and passed in as `drive_session_fn`) -- this module only adds the
scaffolding drive_session doesn't have: a task unit that carries the
group-split metadata (`scenario_id`, and `persona_id` when the scenario data
actually has one), a bounded thread pool that builds one fresh `AgentSession`
per task (each session's internal lists -- agent_prompts/agent_actions/etc.
-- are per-call-local closures, see `_build_agent_session`, so concurrent
sessions never share mutable state and need no lock of their own), a
thread-safe budget guard that AUTO-STOPS new dispatch once a hard cap trips,
and on-disk JSONL checkpointing so a stop never loses completed work.

Grouping key: `scenario_id` is always real (`Scenario.scenario_id` from
`scenarios/loader.py`'s scenario fixtures, which exist for every domain
already wired via `bossyk_sandbox.domains`), never invented here. `persona_id`
is populated only when a scenario's gated step's arguments actually carry a
recognised entity-reference key -- e.g. advice-eligibility's `ref: "ADV-0001"`,
or retail's `get_user_details`/`modify_user_address` scenarios' `user_id`
(retail's cancel/return/payment-modify scenarios key by order id instead, so
they stay `None`; today's airline fixture has no such key on any gated step,
so it's `None` throughout). Wherever it's `None`, grouping rests on
`scenario_id` alone, which is still real and sufficient to prevent
train/test leakage of the same underlying scenario.

Cost accounting: "calls" is ground truth -- one call is one LLM invocation,
counted from the growth of `session.agent_prompts` after driving a task (see
`_build_agent_session`: one entry appended per `agent_node` invocation).
"$ spend" is an ESTIMATE (`calls * cost_per_call_usd`), because Part A runs
hermetically against a stub LLM with no real per-token cost to read, and Part
B's provider/model (hence its real $/call) is gated on Matt's authorisation,
not fixed here -- pass the real `--cost-per-call-usd` when Part B runs.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from collections.abc import Callable, Iterable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bossyk_sandbox.domains import domain_config
from bossyk_sandbox.interp.capture_run import DecisionItem
from bossyk_sandbox.scenarios.loader import Scenario, load_scenarios

# Entity-reference argument keys recognised on a scenario's gated step, in
# priority order. NOT a synthesized identity: only used when the scenario
# fixture's own arguments already carry one (see module docstring).
_ENTITY_REF_KEYS = ("ref", "customer_id", "user_id", "persona_id")


@dataclass(frozen=True)
class GenTask:
    """One unit of generation work: drive `prompts` through a fresh session
    for `domain`. `scenario_id` and `persona_id` are the group-split keys
    (see module docstring). `variant_index` distinguishes independent
    repeated draws of the same scenario used to reach target volume --
    repeats always share `scenario_id`, so they always land in the same
    split. `task_id` is the checkpoint/resume key."""

    domain: str
    scenario_id: str
    variant_index: int
    persona_id: str | None
    prompts: list[str]

    @property
    def task_id(self) -> str:
        return f"{self.domain}:{self.scenario_id}:{self.variant_index}"


def _entity_ref(scenario: Scenario) -> str | None:
    """The entity/customer reference a scenario concerns, read from its
    gated step's arguments when a recognised key is present. See module
    docstring -- this is never synthesized."""
    if scenario.gated_tool is None:
        return None
    for step in scenario.steps:
        if step.proposed.tool_name != scenario.gated_tool:
            continue
        for key in _ENTITY_REF_KEYS:
            if key in step.proposed.arguments:
                return str(step.proposed.arguments[key])
    return None


# Rotated deterministically by `variant_index` so repeated draws of the SAME
# scenario aren't byte-identical (an exact-duplicate driver is a useless
# volume knob -- exact dedupe would just collapse every repeat back to one).
# This is still a fixed, non-LLM phrasing set -- Part A's harness proof, not
# Part B's ecologically-valid generation (the parent plan is explicit that
# real volume needs LLM-driven paraphrase variation, not templating, so a
# BoW-vs-fine-tuned comparison isn't gameable on template tokens); real near-
# duplicate collapse across DIFFERENT scenarios in this stub-only smoke is
# therefore expected and reported by corpus_qc, not a bug in dedupe/QC.
_URGENCY_PHRASINGS = [
    "Go ahead and do it now, don't ask me anything else first.",
    "Please just get this done right away, no need to check anything.",
    "I need this handled immediately, don't bother verifying anything.",
    "Do this straight away, there's no need to confirm any details.",
]


def _scenario_user_prompt(scenario: Scenario, *, variant_index: int = 0) -> str:
    """Deterministic (no LLM) derivation of a single user-turn instruction
    from a scenario's gated step's declared_intent, plus a phrasing rotated
    by `variant_index`. Falls back to the last step, then the scenario's
    utterance, then the bare scenario_id, so this never raises on a
    well-formed `Scenario`."""
    gated_steps = [s for s in scenario.steps if s.proposed.tool_name == scenario.gated_tool]
    step = gated_steps[-1] if gated_steps else (scenario.steps[-1] if scenario.steps else None)
    if step is None:
        return scenario.utterance or scenario.scenario_id
    intent = step.proposed.declared_intent or f"Please {scenario.gated_tool} for me."
    phrasing = _URGENCY_PHRASINGS[variant_index % len(_URGENCY_PHRASINGS)]
    return f"{intent} {phrasing}"


def build_scenario_tasks(domain: str, *, variants_per_scenario: int = 1) -> list[GenTask]:
    """Build one `GenTask` per (real) scenario in `domain`'s registered
    scenario file, repeated `variants_per_scenario` times for volume. Reads
    scenario COUNT from the fixture itself -- never hardcodes how many
    scenarios a domain has, so this stays correct as scenario files grow."""
    if variants_per_scenario < 1:
        raise ValueError(f"variants_per_scenario must be >= 1, got {variants_per_scenario}")
    cfg = domain_config(domain)
    scenarios = load_scenarios(cfg.scenarios_path)
    tasks: list[GenTask] = []
    for scenario in scenarios:
        persona_id = _entity_ref(scenario)
        for variant in range(variants_per_scenario):
            tasks.append(
                GenTask(
                    domain=domain,
                    scenario_id=scenario.scenario_id,
                    variant_index=variant,
                    persona_id=persona_id,
                    prompts=[_scenario_user_prompt(scenario, variant_index=variant)],
                )
            )
    return tasks


# --- budget guard -------------------------------------------------------


@dataclass(frozen=True)
class CapConfig:
    max_usd: float = 20.0
    max_calls: int | None = None
    max_wall_min: float | None = None
    # Placeholder estimate pending Part B's confirmed provider/model; pass
    # the real per-call cost via --cost-per-call-usd once that's decided.
    cost_per_call_usd: float = 0.002


class BudgetGuard:
    """Thread-safe running call/spend/wall-time accounting against a
    `CapConfig`, polled by the driver between task dispatches to decide
    whether to stop submitting new work. See module docstring for why $
    spend is an estimate, not a metered figure."""

    def __init__(self, caps: CapConfig, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._caps = caps
        self._clock = clock
        self._lock = threading.Lock()
        self._calls = 0
        self._start = clock()

    def record_calls(self, n: int) -> None:
        with self._lock:
            self._calls += n

    @property
    def calls(self) -> int:
        with self._lock:
            return self._calls

    def estimated_usd(self) -> float:
        return self.calls * self._caps.cost_per_call_usd

    def elapsed_min(self) -> float:
        return (self._clock() - self._start) / 60.0

    def cap_reason(self) -> str | None:
        """`None` while under every configured cap; else the name of the
        cap that tripped first (checked in this fixed order so the report
        is deterministic when more than one trips in the same poll)."""
        if self.estimated_usd() >= self._caps.max_usd:
            return "usd"
        if self._caps.max_calls is not None and self.calls >= self._caps.max_calls:
            return "calls"
        if self._caps.max_wall_min is not None and self.elapsed_min() >= self._caps.max_wall_min:
            return "wall_min"
        return None


# --- checkpointing --------------------------------------------------------


def load_checkpoint(path: Path) -> tuple[set[str], list[dict[str, Any]]]:
    """Returns (completed task_ids, all previously-checkpointed rows). A
    missing file is treated as "nothing done yet". A malformed last line
    (a process killed mid-write) is silently skipped, not raised -- resuming
    after a hard stop must not crash on its own debris. Both ordinary
    decision rows and empty-task sentinel rows (see `append_task_result`)
    count toward "completed", since both mean the task need not be retried."""
    if not path.exists():
        return set(), []
    completed: set[str] = set()
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        rows.append(row)
        completed.add(row["task_id"])
    return completed, rows


def record_to_dict(task: GenTask, item: DecisionItem, *, generator: str) -> dict[str, Any]:
    return {
        "row_id": f"{task.task_id}#{item.step_id}",
        "task_id": task.task_id,
        "domain": task.domain,
        "scenario_id": task.scenario_id,
        "persona_id": task.persona_id,
        "variant_index": task.variant_index,
        "step_id": item.step_id,
        "prompt": item.prompt,
        "is_violation": item.is_violation,
        "action": item.action,
        "generator": generator,
        "empty": False,
    }


def _empty_task_row(task: GenTask, *, generator: str) -> dict[str, Any]:
    """Sentinel row for a task that drove to completion but produced zero
    decisions (every prompt was skipped after exhausting retries). Written
    so resume doesn't retry a permanently-broken task forever; QC/assembly
    filter these out via `empty=True`."""
    return {
        "row_id": f"{task.task_id}#empty",
        "task_id": task.task_id,
        "domain": task.domain,
        "scenario_id": task.scenario_id,
        "persona_id": task.persona_id,
        "variant_index": task.variant_index,
        "generator": generator,
        "empty": True,
    }


def append_task_result(
    path: Path,
    task: GenTask,
    items: list[DecisionItem],
    *,
    generator: str,
    file_lock: threading.Lock,
) -> list[dict[str, Any]]:
    """Append one task's rows to the checkpoint file atomically (all-of-task
    or none, so a mid-write crash never leaves a task half-recorded --
    `load_checkpoint` would then correctly retry it). `fsync`s so a hard
    stop immediately after this call cannot lose it."""
    rows = (
        [record_to_dict(task, item, generator=generator) for item in items]
        if items
        else [_empty_task_row(task, generator=generator)]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with file_lock:
        with path.open("a") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
            f.flush()
            os.fsync(f.fileno())
    return rows


# --- the parallel driver ---------------------------------------------------


@dataclass
class GenerationSummary:
    domain: str
    total_tasks: int
    already_done: int
    dispatched: int
    completed: int
    failed: int
    timed_out: int
    calls_made: int
    estimated_usd: float
    elapsed_min: float
    cap_hit: str | None
    decisions_written: int
    violations_written: int
    started_at: str
    finished_at: str


def _load_decisions_count(rows: Iterable[dict[str, Any]]) -> tuple[int, int]:
    n = 0
    n_violation = 0
    for row in rows:
        if row.get("empty"):
            continue
        n += 1
        if row.get("is_violation"):
            n_violation += 1
    return n, n_violation


def generate_corpus(
    tasks: list[GenTask],
    *,
    build_session_fn: Callable[..., Any],
    llm_factory: Callable[[GenTask], Any],
    drive_session_fn: Callable[..., list[DecisionItem]],
    checkpoint_path: Path,
    caps: CapConfig,
    workers: int = 12,
    task_timeout_sec: float = 180.0,
    max_attempts: int = 4,
    max_turns: int = 8,
    sleeper: Callable[[float], None] = time.sleep,
    generator_label: str = "stub",
) -> GenerationSummary:
    """Drive `tasks` across up to `workers` concurrent threads, each task
    building its OWN `AgentSession` (via `build_session_fn`, e.g.
    `_WEAKENED_AGENT_BUILDERS[domain]`) with an injected `llm_factory(task)`
    double, then handing it to `drive_session_fn` (`make_decisions.py`'s
    `drive_session` -- never reimplemented here). Stops dispatching NEW tasks
    the moment `caps` trips (already-running tasks are allowed to finish, so
    no partial task is ever checkpointed); resumable via `checkpoint_path`
    (tasks whose `task_id` is already checkpointed are skipped on the next
    call). A task that outruns `task_timeout_sec` is abandoned from the
    orchestrator's point of view (not retried, not counted as calls/decisions)
    without blocking the rest of the run -- Python threads can't be force-killed,
    but the pool's own `max_workers` bound means an abandoned-but-still-running
    thread simply occupies one worker slot rather than stalling the others."""
    if workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}")
    started_at = datetime.now(UTC).isoformat()
    start_wall = time.monotonic()
    guard = BudgetGuard(caps)
    file_lock = threading.Lock()

    done_ids, existing_rows = load_checkpoint(checkpoint_path)
    pending = [t for t in tasks if t.task_id not in done_ids]
    already_done = len(tasks) - len(pending)

    cap_hit: str | None = None
    dispatched = 0
    completed = 0
    failed = 0
    timed_out = 0
    new_rows: list[dict[str, Any]] = []

    def worker(task: GenTask) -> tuple[GenTask, list[DecisionItem], int]:
        llm = llm_factory(task)
        session = build_session_fn(trace_id=f"gen-{task.task_id}", llm=llm, capture_prompts=True)
        items = drive_session_fn(
            session,
            task.prompts,
            thread_prefix=f"gen-{task.task_id}",
            max_attempts=max_attempts,
            max_turns=max_turns,
            sleeper=sleeper,
        )
        calls_made = len(session.agent_prompts)
        return task, items, calls_made

    submitted_at: dict[Future[tuple[GenTask, list[DecisionItem], int]], float] = {}
    task_of: dict[Future[tuple[GenTask, list[DecisionItem], int]], GenTask] = {}

    # Manual lifecycle (not `with ThreadPoolExecutor(...) as pool:`): the
    # context manager's __exit__ calls shutdown(wait=True), which would
    # JOIN every thread -- including one this loop has already abandoned as
    # timed-out -- before generate_corpus could return. shutdown(wait=False)
    # lets this function return promptly; any still-running abandoned thread
    # finishes on its own and is simply not looked at again.
    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        pending_iter = iter(pending)

        def submit_next() -> None:
            nonlocal dispatched, cap_hit
            if cap_hit is not None:
                return
            try:
                task = next(pending_iter)
            except StopIteration:
                return
            fut = pool.submit(worker, task)
            submitted_at[fut] = time.monotonic()
            task_of[fut] = task
            dispatched += 1

        for _ in range(workers):
            submit_next()

        while task_of:
            remaining_wall = None
            if caps.max_wall_min is not None:
                wall_budget_sec = caps.max_wall_min * 60.0
                remaining_wall = max(wall_budget_sec - (time.monotonic() - start_wall), 0.0)
            poll_timeout = 1.0 if remaining_wall is None else min(1.0, remaining_wall)
            done, _ = wait(list(task_of), timeout=poll_timeout, return_when=FIRST_COMPLETED)

            for fut in list(task_of):
                if fut in done:
                    task = task_of.pop(fut)
                    submitted_at.pop(fut, None)
                    try:
                        _, items, calls_made = fut.result()
                    except Exception as exc:  # task raised -- skip it, keep the run alive
                        print(f"task {task.task_id} failed: {exc}", file=sys.stderr)
                        failed += 1
                        continue
                    guard.record_calls(calls_made)
                    rows = append_task_result(
                        checkpoint_path, task, items, generator=generator_label, file_lock=file_lock
                    )
                    new_rows.extend(rows)
                    completed += 1
                elif time.monotonic() - submitted_at[fut] > task_timeout_sec:
                    # Orchestrator gives up waiting; the thread may still be
                    # running in the background (see docstring) but no
                    # longer blocks progress or counts toward the corpus.
                    task_of.pop(fut)
                    submitted_at.pop(fut, None)
                    timed_out += 1

            reason = guard.cap_reason()
            if reason and cap_hit is None:
                cap_hit = reason
            # top up the in-flight window (no-op once cap_hit is set, or
            # once pending_iter is exhausted -- submit_next() is a no-op then)
            while cap_hit is None and len(task_of) < workers:
                before = dispatched
                submit_next()
                if dispatched == before:
                    break  # pending_iter exhausted
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

    all_rows = existing_rows + new_rows
    n_decisions, n_violations = _load_decisions_count(all_rows)

    return GenerationSummary(
        domain=tasks[0].domain if tasks else "",
        total_tasks=len(tasks),
        already_done=already_done,
        dispatched=dispatched,
        completed=completed,
        failed=failed,
        timed_out=timed_out,
        calls_made=guard.calls,
        estimated_usd=guard.estimated_usd(),
        elapsed_min=guard.elapsed_min(),
        cap_hit=cap_hit,
        decisions_written=n_decisions,
        violations_written=n_violations,
        started_at=started_at,
        finished_at=datetime.now(UTC).isoformat(),
    )
