"""Part A item 4: SMOKE the entire datagen pipeline at the REAL worker count
(~12-13) with a STUB agent (phase-detector-training-step2-datagen.md).

ZERO API calls: every LLM here is `generate_corpus.StubScenarioLLM`, which
replays each scenario's own authored tool-call sequence (imported from the
script by path, same convention as `test_make_decisions_script.py`) -- no
network, no API key, nothing billable.

Exercises, end to end, across all three domains (retail, airline,
advice-eligibility):
    generate (capped -> proves auto-stop) -> resume (proves no lost/duplicate
    work) -> assemble (dedupe + stratified group-split) -> QC (leakage scan,
    cross-domain near-dup scan, class balance, hand-audit sample, manifest)

This is the un-gated proof the harness is sound before Part B (billable) is
ever authorized to run.
"""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path
from types import ModuleType

from bossyk_sandbox.interp.corpus_assembly import (
    dedupe,
    load_decisions_jsonl,
    stratified_group_split,
)
from bossyk_sandbox.interp.corpus_qc import (
    build_manifest,
    class_balance_report,
    cross_domain_near_duplicate_scan,
    export_hand_audit_sample,
    leakage_scan,
)
from bossyk_sandbox.interp.datagen_driver import CapConfig, build_scenario_tasks, generate_corpus

GENERATE_SCRIPT_PATH = Path(__file__).parent.parent.parent / "scripts" / "generate_corpus.py"
MAKE_DECISIONS_SCRIPT_PATH = Path(__file__).parent.parent.parent / "scripts" / "make_decisions.py"

DOMAINS = ("retail", "airline", "advice-eligibility")
REAL_WORKER_COUNT = 13  # ~80% of the box's 16 physical cores, per the plan
# Matches len(datagen_driver._URGENCY_PHRASINGS) exactly: that phrasing pool
# rotates by variant_index % len(pool), so variants_per_scenario beyond the
# pool size would repeat a phrasing and (same scenario + same phrasing) become
# an EXACT text duplicate -- correctly caught by dedupe_exact, but that would
# make this smoke's "zero drops expected" assertion domain-fixture-dependent
# instead of a clean proof. Staying at the pool size sidesteps that.
VARIANTS_PER_SCENARIO = 4


def _import_by_path(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Registering in sys.modules BEFORE exec_module matters here (unlike the
    # simpler by-path imports elsewhere in this test suite): generate_corpus.py
    # defines a `@dataclass` (StubScenarioLLM) under `from __future__ import
    # annotations`, and dataclass's string-annotation resolution looks the
    # module up via `sys.modules[cls.__module__]` -- without this line that
    # lookup returns None and dataclass() raises AttributeError.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_full_pipeline_smoke_at_real_worker_count_zero_api_calls(tmp_path: Path) -> None:
    generate_corpus_script = _import_by_path("generate_corpus_smoke", GENERATE_SCRIPT_PATH)
    make_decisions = _import_by_path("make_decisions_smoke", MAKE_DECISIONS_SCRIPT_PATH)

    # ---- 1. build tasks across all three transfer domains, real scenario counts ----
    tasks_by_domain = {
        domain: build_scenario_tasks(domain, variants_per_scenario=VARIANTS_PER_SCENARIO)
        for domain in DOMAINS
    }
    total_tasks = sum(len(t) for t in tasks_by_domain.values())
    print(f"\n[smoke] worker_count={REAL_WORKER_COUNT} total_tasks={total_tasks}")
    assert total_tasks > REAL_WORKER_COUNT * 2  # enough volume to actually saturate the pool

    checkpoints = {domain: tmp_path / domain / "decisions.jsonl" for domain in DOMAINS}
    summaries_first = {}
    start = time.monotonic()

    # ---- 2. capped first pass: prove auto-stop actually fires ----
    for domain in DOMAINS:
        llm_factory = generate_corpus_script.stub_llm_factory_for_domain(domain)
        summary = generate_corpus(
            tasks_by_domain[domain],
            build_session_fn=generate_corpus_script._BUILDERS[domain],
            llm_factory=llm_factory,
            drive_session_fn=make_decisions.drive_session,
            checkpoint_path=checkpoints[domain],
            caps=CapConfig(max_usd=1000.0, max_calls=6),  # small enough to trip before done
            workers=REAL_WORKER_COUNT,
        )
        summaries_first[domain] = summary
        print(
            f"[smoke] pass1 {domain}: cap_hit={summary.cap_hit} completed={summary.completed}"
            f"/{summary.total_tasks} calls={summary.calls_made}"
        )
        assert summary.cap_hit == "calls"  # the cap ACTUALLY fired
        assert summary.completed < summary.total_tasks  # ...and stopped short of done

    # ---- 3. resume with no cap: prove nothing completed is lost or duplicated ----
    summaries_second = {}
    for domain in DOMAINS:
        llm_factory = generate_corpus_script.stub_llm_factory_for_domain(domain)
        summary = generate_corpus(
            tasks_by_domain[domain],
            build_session_fn=generate_corpus_script._BUILDERS[domain],
            llm_factory=llm_factory,
            drive_session_fn=make_decisions.drive_session,
            checkpoint_path=checkpoints[domain],
            caps=CapConfig(max_usd=1000.0),
            workers=REAL_WORKER_COUNT,
        )
        summaries_second[domain] = summary
        print(
            f"[smoke] pass2(resume) {domain}: already_done={summary.already_done} "
            f"completed={summary.completed} cap_hit={summary.cap_hit}"
        )
        assert summary.cap_hit is None
        assert summary.already_done == summaries_first[domain].completed
        assert summary.already_done + summary.completed == len(tasks_by_domain[domain])

        completed_ids, rows = _load_checkpoint_ids(checkpoints[domain])
        assert len(completed_ids) == len(tasks_by_domain[domain])
        row_ids = [r["row_id"] for r in rows if not r.get("empty")]
        assert len(row_ids) == len(set(row_ids))  # no duplicate decision rows across the two passes

    elapsed = time.monotonic() - start
    print(f"[smoke] generation (both passes, {len(DOMAINS)} domains) elapsed={elapsed:.2f}s")

    # ---- 4. assemble: dedupe + stratified group-split, per domain ----
    all_records_by_domain = {}
    splits_by_domain = {}
    dedupe_reports = {}
    for domain in DOMAINS:
        records = load_decisions_jsonl(checkpoints[domain])
        deduped, dedupe_report = dedupe(records)
        dedupe_reports[domain] = dedupe_report
        all_records_by_domain[domain] = deduped
        splits, split_report = stratified_group_split(
            deduped, val_frac=0.15, test_frac=0.15, seed=0
        )
        splits_by_domain[domain] = splits
        print(
            f"[smoke] assemble {domain}: n_input={dedupe_report.n_input} "
            f"exact_dropped={dedupe_report.n_exact_dropped} "
            f"near_dropped={dedupe_report.n_near_dropped} n_output={dedupe_report.n_output} "
            f"n_groups={split_report.n_groups} group_fields={split_report.group_fields}"
        )
        # Zero exact/near-dup collapse expected: variant phrasing differs and
        # near-dup comparison already strips the shared policy boilerplate
        # (see corpus_assembly.near_duplicate_comparison_text).
        assert dedupe_report.n_output == dedupe_report.n_input
        for rows in splits.values():
            assert all(r["domain"] == domain for r in rows)

    # ---- 5. QC: leakage scan, cross-domain near-dup, class balance, audit sample, manifest ----
    combined_splits: dict[str, list[dict[str, object]]] = {"train": [], "val": [], "test": []}
    all_records: list[dict[str, object]] = []
    for domain in DOMAINS:
        for split_name, rows in splits_by_domain[domain].items():
            combined_splits[split_name].extend(rows)
        all_records.extend(all_records_by_domain[domain])

    leak_report = leakage_scan(combined_splits, group_fields=("scenario_id", "persona_id"))
    print(
        f"[smoke] leakage scan: clean={leak_report.clean} leaked={len(leak_report.leaked_groups)}"
    )
    assert leak_report.clean

    cross_domain_report = cross_domain_near_duplicate_scan(all_records)
    print(
        f"[smoke] cross-domain near-dup: {len(cross_domain_report.pairs)}/"
        f"{cross_domain_report.n_pairs_checked} pairs, rate={cross_domain_report.rate:.4f}"
    )
    assert (
        cross_domain_report.rate == 0.0
    )  # retail/airline/advice-eligibility stay lexically distinct

    balance = class_balance_report(combined_splits)
    for domain in DOMAINS:
        print(f"[smoke] class balance {domain}: {balance.get(domain)}")
        # every domain produced BOTH classes overall (train has the volume to guarantee this)
        assert balance[domain]["train"]["violations"] > 0
        assert balance[domain]["train"]["violations"] < balance[domain]["train"]["n"]

    for domain in DOMAINS:
        domain_records = [r for r in all_records if r["domain"] == domain]
        sample = export_hand_audit_sample(domain_records, n=50, seed=0)
        assert len(sample) == min(50, len(domain_records))
        manifest = build_manifest(
            domain=domain,
            generator_model="stub-scenario-replay",
            serving_path="stub",
            seed=0,
            counts={s: len(splits_by_domain[domain][s]) for s in ("train", "val", "test")},
            class_balance={
                s: balance.get(domain, {}).get(s, {}).get("violation_rate", 0.0)
                for s in ("train", "val", "test")
            },
            cap_hit=None,  # the FINAL (resumed) pass hit no cap
            near_dup_rate_within_domain=(
                dedupe_reports[domain].n_near_dropped / dedupe_reports[domain].n_input
                if dedupe_reports[domain].n_input
                else 0.0
            ),
            generated_at="smoke-test",
        )
        print(f"[smoke] manifest {domain}: {manifest}")
        assert manifest["cap_was_hit"] is False
        assert manifest["domain"] == domain


def _load_checkpoint_ids(path: Path) -> tuple[set[str], list[dict[str, object]]]:
    from bossyk_sandbox.interp.datagen_driver import load_checkpoint

    return load_checkpoint(path)
