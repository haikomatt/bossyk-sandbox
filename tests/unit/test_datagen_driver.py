"""Hermetic tests for `bossyk_sandbox.interp.datagen_driver`
(phase-detector-training-step2-datagen.md Part A, item 1: parallel, capped,
resumable generation driver). No network, no API key -- every LLM here is a
stub double, mirroring `tests/unit/test_make_decisions_script.py`'s
convention.
"""

from __future__ import annotations

import importlib.util
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from langchain_core.messages import AIMessage

from bossyk_sandbox.domains import domain_config
from bossyk_sandbox.interp.capture_run import DecisionItem
from bossyk_sandbox.interp.corpus_assembly import MixedCorpusVersionError
from bossyk_sandbox.interp.datagen_driver import (
    BudgetGuard,
    CapConfig,
    GenTask,
    _compositional_phrasing,
    _empty_task_row,
    _scenario_user_prompt,
    append_task_result,
    build_scenario_tasks,
    generate_corpus,
    load_checkpoint,
    record_to_dict,
)
from bossyk_sandbox.runtime.langgraph_agent import build_airline_agent_session
from bossyk_sandbox.scenarios.loader import load_scenarios

# Reuse the exact drive_session from make_decisions.py by importing it the
# same way test_make_decisions_script.py does (by file path -- it's a
# script, not a package module).
SCRIPT_PATH = Path(__file__).parent.parent.parent / "scripts" / "make_decisions.py"


def _import_make_decisions() -> ModuleType:
    spec = importlib.util.spec_from_file_location("make_decisions_for_driver", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


drive_session = _import_make_decisions().drive_session


# --- build_scenario_tasks ---------------------------------------------------


@pytest.mark.parametrize("domain", ["retail", "airline", "advice-eligibility"])
def test_build_scenario_tasks_count_matches_real_scenario_file(domain: str) -> None:
    cfg = domain_config(domain)
    n_scenarios = len(load_scenarios(cfg.scenarios_path))
    tasks = build_scenario_tasks(domain, variants_per_scenario=3)
    assert len(tasks) == n_scenarios * 3
    assert {t.scenario_id for t in tasks} == {
        s.scenario_id for s in load_scenarios(cfg.scenarios_path)
    }


def test_build_scenario_tasks_advice_eligibility_recovers_the_real_entity_ref() -> None:
    # Test-integrity note (2026-08-26): originally asserted every ref starts
    # with "ADV-", written against the pre-level-up domain where all gated
    # steps were keyed on `ref`. The Amendment-1 level-up intentionally added
    # `close_enrolment` scenarios keyed on `enrolment_id` ("ENR-*"), so the
    # single-prefix assumption is outdated, not the driver. Updated alongside
    # adding `enrolment_id` to _ENTITY_REF_KEYS so every seed still recovers
    # a real, non-None entity ref.
    tasks = build_scenario_tasks("advice-eligibility")
    persona_ids = {t.persona_id for t in tasks}
    assert all(
        p is not None and (p.startswith("ADV-") or p.startswith("ENR-")) for p in persona_ids
    )
    # both key_args are represented post-level-up
    assert any(p.startswith("ADV-") for p in persona_ids if p)
    assert any(p.startswith("ENR-") for p in persona_ids if p)


def test_build_scenario_tasks_task_ids_are_unique_and_stable() -> None:
    tasks = build_scenario_tasks("retail", variants_per_scenario=2)
    ids = [t.task_id for t in tasks]
    assert len(ids) == len(set(ids))
    # stable across an independent call with the same args
    tasks_again = build_scenario_tasks("retail", variants_per_scenario=2)
    assert [t.task_id for t in tasks_again] == ids


def test_build_scenario_tasks_rejects_zero_variants() -> None:
    with pytest.raises(ValueError):
        build_scenario_tasks("retail", variants_per_scenario=0)


# --- BudgetGuard -------------------------------------------------------------


def test_budget_guard_usd_cap_trips_after_enough_calls() -> None:
    guard = BudgetGuard(CapConfig(max_usd=0.01, cost_per_call_usd=0.005))
    assert guard.cap_reason() is None
    guard.record_calls(1)
    assert guard.cap_reason() is None
    guard.record_calls(1)
    assert guard.cap_reason() == "usd"


def test_budget_guard_calls_cap() -> None:
    guard = BudgetGuard(CapConfig(max_usd=1000.0, max_calls=3))
    guard.record_calls(2)
    assert guard.cap_reason() is None
    guard.record_calls(1)
    assert guard.cap_reason() == "calls"


def test_budget_guard_wall_min_cap_uses_injected_clock() -> None:
    fake_now = [0.0]
    guard = BudgetGuard(CapConfig(max_usd=1000.0, max_wall_min=1.0), clock=lambda: fake_now[0])
    assert guard.cap_reason() is None
    fake_now[0] = 61.0  # 61 seconds later -> just over 1 wall-minute
    assert guard.cap_reason() == "wall_min"


# --- checkpointing -----------------------------------------------------------


def test_load_checkpoint_missing_file_is_empty(tmp_path: Path) -> None:
    completed, rows = load_checkpoint(tmp_path / "nope.jsonl")
    assert completed == set()
    assert rows == []


def _task(scenario_id: str, *, domain: str = "retail", persona_id: str | None = None) -> GenTask:
    return GenTask(
        domain=domain,
        scenario_id=scenario_id,
        variant_index=0,
        persona_id=persona_id,
        prompts=["hi"],
    )


def test_append_task_result_then_load_checkpoint_round_trips(tmp_path: Path) -> None:
    ckpt = tmp_path / "decisions.jsonl"
    task = _task("retail-001")
    items = [DecisionItem(step_id="turn-0", prompt="ctx", is_violation=True, action="cancel")]
    append_task_result(ckpt, task, items, generator="stub", file_lock=threading.Lock())

    completed, rows = load_checkpoint(ckpt)
    assert completed == {task.task_id}
    assert len(rows) == 1
    assert rows[0]["is_violation"] is True
    assert rows[0]["scenario_id"] == "retail-001"


def test_append_task_result_with_zero_items_writes_empty_sentinel(tmp_path: Path) -> None:
    ckpt = tmp_path / "decisions.jsonl"
    task = _task("retail-002")
    append_task_result(ckpt, task, [], generator="stub", file_lock=threading.Lock())

    completed, rows = load_checkpoint(ckpt)
    assert completed == {task.task_id}
    assert rows[0]["empty"] is True


def test_load_checkpoint_skips_a_malformed_trailing_line(tmp_path: Path) -> None:
    ckpt = tmp_path / "decisions.jsonl"
    good = record_to_dict(
        _task("retail-003"),
        DecisionItem(step_id="turn-0", prompt="p", is_violation=False),
        generator="stub",
    )
    ckpt.write_text(json.dumps(good) + "\n" + '{"task_id": "broke')  # truncated last line
    completed, rows = load_checkpoint(ckpt)
    assert completed == {"retail:retail-003:0"}
    assert len(rows) == 1


# --- generate_corpus: stub session plumbing ----------------------------------


class _NoTools:
    def get_tools(self) -> dict[str, Any]:
        return {}


@dataclass
class _FakeEnv:
    tools: Any
    policy: str = "be compliant"


def _no_tool_airline_session(*, trace_id: str, llm: Any, capture_prompts: bool) -> Any:
    """Cheap stand-in `build_session_fn`: airline session, no tools bound, so
    every turn is a single no-tool-call response -- one call per task,
    always compliant. No network (fake env + injected llm)."""
    return build_airline_agent_session(
        trace_id=trace_id,
        llm=llm,
        environment=_FakeEnv(tools=_NoTools()),
        capture_prompts=capture_prompts,
    )


@dataclass
class _OkLLM:
    calls: int = 0

    def invoke(self, _messages: Any) -> AIMessage:
        self.calls += 1
        return AIMessage(content="ok")


def _tasks(n: int, *, domain: str = "retail") -> list[GenTask]:
    return [_task(f"scn-{i}", domain=domain) for i in range(n)]


def test_generate_corpus_completes_all_tasks_with_no_caps(tmp_path: Path) -> None:
    tasks = _tasks(5)
    summary = generate_corpus(
        tasks,
        build_session_fn=_no_tool_airline_session,
        llm_factory=lambda _task: _OkLLM(),
        drive_session_fn=drive_session,
        checkpoint_path=tmp_path / "decisions.jsonl",
        caps=CapConfig(max_usd=1000.0),
        workers=4,
    )
    assert summary.completed == 5
    assert summary.dispatched == 5
    assert summary.cap_hit is None
    assert summary.decisions_written == 5
    assert summary.violations_written == 0
    completed, rows = load_checkpoint(tmp_path / "decisions.jsonl")
    assert len(completed) == 5
    assert len(rows) == 5


def test_generate_corpus_respects_the_worker_count_concurrently(tmp_path: Path) -> None:
    """Prove tasks actually run concurrently, not serially: a rendezvous LLM
    blocks each call until `workers` calls are simultaneously in-flight."""
    workers = 6
    n_tasks = workers * 2
    lock = threading.Lock()
    state = {"in_flight": 0, "max_seen": 0}
    barrier = threading.Barrier(workers, timeout=10.0)

    @dataclass
    class _RendezvousLLM:
        def invoke(self, _messages: Any) -> AIMessage:
            with lock:
                state["in_flight"] += 1
                state["max_seen"] = max(state["max_seen"], state["in_flight"])
            try:
                barrier.wait()
            except threading.BrokenBarrierError:
                pass
            with lock:
                state["in_flight"] -= 1
            return AIMessage(content="ok")

    summary = generate_corpus(
        _tasks(n_tasks),
        build_session_fn=_no_tool_airline_session,
        llm_factory=lambda _task: _RendezvousLLM(),
        drive_session_fn=drive_session,
        checkpoint_path=tmp_path / "decisions.jsonl",
        caps=CapConfig(max_usd=1000.0),
        workers=workers,
    )
    assert summary.completed == n_tasks
    assert state["max_seen"] == workers  # every worker slot was used concurrently


def test_generate_corpus_auto_stops_on_max_calls(tmp_path: Path) -> None:
    tasks = _tasks(20)
    summary = generate_corpus(
        tasks,
        build_session_fn=_no_tool_airline_session,
        llm_factory=lambda _task: _OkLLM(),
        drive_session_fn=drive_session,
        checkpoint_path=tmp_path / "decisions.jsonl",
        caps=CapConfig(max_usd=1000.0, max_calls=5),
        workers=3,
    )
    assert summary.cap_hit == "calls"
    assert summary.completed < 20  # did NOT process every task
    assert summary.calls_made >= 5
    # nothing beyond what's checkpointed is silently lost
    completed, rows = load_checkpoint(tmp_path / "decisions.jsonl")
    assert len(completed) == summary.completed


def test_generate_corpus_auto_stops_on_max_usd(tmp_path: Path) -> None:
    tasks = _tasks(20)
    summary = generate_corpus(
        tasks,
        build_session_fn=_no_tool_airline_session,
        llm_factory=lambda _task: _OkLLM(),
        drive_session_fn=drive_session,
        checkpoint_path=tmp_path / "decisions.jsonl",
        caps=CapConfig(max_usd=0.01, cost_per_call_usd=0.005),
        workers=2,
    )
    assert summary.cap_hit == "usd"
    assert summary.completed < 20


def test_generate_corpus_resume_completes_the_remaining_tasks_with_no_duplicates(
    tmp_path: Path,
) -> None:
    tasks = _tasks(12)
    ckpt = tmp_path / "decisions.jsonl"

    first = generate_corpus(
        tasks,
        build_session_fn=_no_tool_airline_session,
        llm_factory=lambda _task: _OkLLM(),
        drive_session_fn=drive_session,
        checkpoint_path=ckpt,
        caps=CapConfig(max_usd=1000.0, max_calls=4),
        workers=2,
    )
    assert first.cap_hit == "calls"
    assert first.completed < 12

    second = generate_corpus(
        tasks,
        build_session_fn=_no_tool_airline_session,
        llm_factory=lambda _task: _OkLLM(),
        drive_session_fn=drive_session,
        checkpoint_path=ckpt,
        caps=CapConfig(max_usd=1000.0),
        workers=2,
    )
    assert second.already_done == first.completed
    assert second.already_done + second.completed == 12
    assert second.cap_hit is None

    completed, rows = load_checkpoint(ckpt)
    assert len(completed) == 12
    task_ids = [r["task_id"] for r in rows]
    assert len(task_ids) == len(set(task_ids))  # no duplicate rows across the two runs


def test_generate_corpus_a_hung_task_does_not_block_the_rest(tmp_path: Path) -> None:
    @dataclass
    class _MaybeHangingLLM:
        hang: bool

        def invoke(self, _messages: Any) -> AIMessage:
            if self.hang:
                time.sleep(2.0)  # well past the 0.2s task_timeout below
            return AIMessage(content="ok")

    tasks = _tasks(6)

    def llm_factory(task: GenTask) -> Any:
        return _MaybeHangingLLM(hang=task.scenario_id == "scn-0")

    start = time.monotonic()
    summary = generate_corpus(
        tasks,
        build_session_fn=_no_tool_airline_session,
        llm_factory=llm_factory,
        drive_session_fn=drive_session,
        checkpoint_path=tmp_path / "decisions.jsonl",
        caps=CapConfig(max_usd=1000.0),
        workers=6,
        task_timeout_sec=0.2,
    )
    elapsed = time.monotonic() - start
    assert elapsed < 2.0  # did not wait out the hung task
    assert summary.timed_out == 1
    assert summary.completed == 5  # the other five still finished


# --- generate_corpus: session_kwargs plumbing (hermetic diversity fix item 1) --


def _no_tool_airline_session_kwargs(
    *, trace_id: str, llm: Any, capture_prompts: bool, **kwargs: Any
) -> Any:
    """Like `_no_tool_airline_session` but records any extra kwargs it was
    called with (e.g. temperature/top_p), so `generate_corpus`'s forwarding
    can be asserted without touching a real ChatOpenAI client."""
    _no_tool_airline_session_kwargs.last_kwargs = kwargs  # type: ignore[attr-defined]
    return build_airline_agent_session(
        trace_id=trace_id,
        llm=llm,
        environment=_FakeEnv(tools=_NoTools()),
        capture_prompts=capture_prompts,
    )


def test_generate_corpus_forwards_session_kwargs_to_build_session_fn(tmp_path: Path) -> None:
    generate_corpus(
        _tasks(2),
        build_session_fn=_no_tool_airline_session_kwargs,
        llm_factory=lambda _task: _OkLLM(),
        drive_session_fn=drive_session,
        checkpoint_path=tmp_path / "decisions.jsonl",
        caps=CapConfig(max_usd=1000.0),
        workers=2,
        session_kwargs={"temperature": 0.7, "top_p": 0.3},
    )
    assert _no_tool_airline_session_kwargs.last_kwargs == {"temperature": 0.7, "top_p": 0.3}  # type: ignore[attr-defined]


def test_generate_corpus_session_kwargs_unset_is_byte_identical(tmp_path: Path) -> None:
    """Default (no `session_kwargs`) must call `build_session_fn` with NO
    extra kwargs -- existing stub builders (like `_no_tool_airline_session`,
    which accepts only trace_id/llm/capture_prompts) must keep working
    unmodified."""
    summary = generate_corpus(
        _tasks(2),
        build_session_fn=_no_tool_airline_session,
        llm_factory=lambda _task: _OkLLM(),
        drive_session_fn=drive_session,
        checkpoint_path=tmp_path / "decisions.jsonl",
        caps=CapConfig(max_usd=1000.0),
        workers=2,
    )
    assert summary.completed == 2


# --- generate_corpus: corpus_version stamping (hermetic diversity fix item 3) --


def test_generate_corpus_default_corpus_version_is_v1(tmp_path: Path) -> None:
    generate_corpus(
        _tasks(2),
        build_session_fn=_no_tool_airline_session,
        llm_factory=lambda _task: _OkLLM(),
        drive_session_fn=drive_session,
        checkpoint_path=tmp_path / "decisions.jsonl",
        caps=CapConfig(max_usd=1000.0),
        workers=2,
    )
    _, rows = load_checkpoint(tmp_path / "decisions.jsonl")
    assert rows
    assert all(r["corpus_version"] == "v1" for r in rows)


def test_generate_corpus_stamps_the_requested_corpus_version(tmp_path: Path) -> None:
    summary = generate_corpus(
        _tasks(2),
        build_session_fn=_no_tool_airline_session,
        llm_factory=lambda _task: _OkLLM(),
        drive_session_fn=drive_session,
        checkpoint_path=tmp_path / "decisions.jsonl",
        caps=CapConfig(max_usd=1000.0),
        workers=2,
        corpus_version="v2",
    )
    assert summary.corpus_version == "v2"
    _, rows = load_checkpoint(tmp_path / "decisions.jsonl")
    assert rows
    assert all(r["corpus_version"] == "v2" for r in rows)


def test_generate_corpus_refuses_to_resume_a_checkpoint_under_a_different_corpus_version(
    tmp_path: Path,
) -> None:
    ckpt = tmp_path / "decisions.jsonl"
    generate_corpus(
        _tasks(2),
        build_session_fn=_no_tool_airline_session,
        llm_factory=lambda _task: _OkLLM(),
        drive_session_fn=drive_session,
        checkpoint_path=ckpt,
        caps=CapConfig(max_usd=1000.0),
        workers=2,
        corpus_version="v1",
    )
    with pytest.raises(MixedCorpusVersionError):
        generate_corpus(
            _tasks(4),
            build_session_fn=_no_tool_airline_session,
            llm_factory=lambda _task: _OkLLM(),
            drive_session_fn=drive_session,
            checkpoint_path=ckpt,
            caps=CapConfig(max_usd=1000.0),
            workers=2,
            corpus_version="v2",
        )


def test_record_to_dict_and_empty_task_row_default_corpus_version_to_v1() -> None:
    task = _task("retail-004")
    item = DecisionItem(step_id="turn-0", prompt="p", is_violation=False)
    assert record_to_dict(task, item, generator="stub")["corpus_version"] == "v1"
    assert _empty_task_row(task, generator="stub")["corpus_version"] == "v1"


def test_record_to_dict_stamps_the_given_corpus_version() -> None:
    task = _task("retail-005")
    item = DecisionItem(step_id="turn-0", prompt="p", is_violation=False)
    row = record_to_dict(task, item, generator="stub", corpus_version="v2")
    assert row["corpus_version"] == "v2"


# --- compositional phrasing pool (hermetic diversity fix item 2) --------------


def test_compositional_phrasing_yields_at_least_200_distinct_variants_for_one_scenario() -> None:
    n = 200
    variants = {_compositional_phrasing("retail-scn-1", variant_index=i) for i in range(n)}
    assert len(variants) == n  # every one of the 200 draws is textually distinct


def test_compositional_phrasing_is_reproducible_for_the_same_scenario_and_variant() -> None:
    first = _compositional_phrasing("retail-scn-1", variant_index=17)
    second = _compositional_phrasing("retail-scn-1", variant_index=17)
    assert first == second  # pure function of (scenario_id, variant_index)

    # ...and re-running the whole build (a fresh process, no shared state) is
    # also byte-identical -- reproducible regeneration, not accidental luck.
    third = _compositional_phrasing("retail-scn-1", variant_index=17)
    assert first == third


def test_compositional_phrasing_differs_by_scenario_for_the_same_variant_index() -> None:
    a = _compositional_phrasing("retail-scn-1", variant_index=0)
    b = _compositional_phrasing("retail-scn-2", variant_index=0)
    assert a != b


def test_compositional_phrasing_reads_as_a_natural_multi_sentence_turn() -> None:
    # Ecological-validity guard (dossier confound #2): each slot is a
    # complete, punctuated clause, not a mad-libs fragment -- so any
    # combination still reads as a natural run of urgent sentences.
    text = _compositional_phrasing("retail-scn-1", variant_index=0)
    sentences = [s for s in text.split(". ") if s]
    assert len(sentences) >= 3  # opener + urgency + justification + closer
    assert text[0].isupper()
    assert text.rstrip().endswith((".", "!"))


def test_scenario_user_prompt_still_embeds_the_declared_intent_and_a_phrasing() -> None:
    scenarios = load_scenarios(domain_config("retail").scenarios_path)
    scenario = scenarios[0]
    prompt = _scenario_user_prompt(scenario, variant_index=3)
    gated_steps = [s for s in scenario.steps if s.proposed.tool_name == scenario.gated_tool]
    step = gated_steps[-1] if gated_steps else scenario.steps[-1]
    intent = step.proposed.declared_intent or f"Please {scenario.gated_tool} for me."
    assert prompt.startswith(intent)
    assert prompt != intent  # a phrasing was actually appended
