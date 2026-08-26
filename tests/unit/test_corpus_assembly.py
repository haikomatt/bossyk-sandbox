"""Hermetic tests for `bossyk_sandbox.interp.corpus_assembly`
(phase-detector-training-step2-datagen.md Part A, item 2: dedupe + stratified
group-split). Pure/offline -- no network, no LLM at all.
"""

from __future__ import annotations

import json
from pathlib import Path

from bossyk_sandbox.interp.corpus_assembly import (
    dedupe,
    dedupe_exact,
    dedupe_near_duplicates,
    default_group_fields,
    load_decisions_jsonl,
    near_duplicate_comparison_text,
    stratified_group_split,
)


def _row(
    scenario_id: str,
    prompt: str,
    *,
    domain: str = "retail",
    is_violation: bool = False,
    persona_id: str | None = None,
    variant_index: int = 0,
) -> dict[str, object]:
    return {
        "row_id": f"{domain}:{scenario_id}:{variant_index}#turn-0",
        "task_id": f"{domain}:{scenario_id}:{variant_index}",
        "domain": domain,
        "scenario_id": scenario_id,
        "persona_id": persona_id,
        "variant_index": variant_index,
        "step_id": "turn-0",
        "prompt": prompt,
        "is_violation": is_violation,
        "action": None,
        "generator": "stub",
        "empty": False,
    }


# --- load_decisions_jsonl ----------------------------------------------


def test_load_decisions_jsonl_missing_file_is_empty(tmp_path: Path) -> None:
    assert load_decisions_jsonl(tmp_path / "nope.jsonl") == []


def test_load_decisions_jsonl_filters_out_empty_sentinel_rows(tmp_path: Path) -> None:
    path = tmp_path / "decisions.jsonl"
    good = _row("retail-001", "cancel my order now")
    empty_sentinel = {"task_id": "retail:retail-002:0", "empty": True}
    path.write_text(json.dumps(good) + "\n" + json.dumps(empty_sentinel) + "\n")
    rows = load_decisions_jsonl(path)
    assert len(rows) == 1
    assert rows[0]["scenario_id"] == "retail-001"


# --- dedupe --------------------------------------------------------------


def test_dedupe_exact_drops_identical_prompt_within_domain() -> None:
    records = [
        _row("s1", "cancel my order now"),
        _row("s2", "cancel my order now"),  # exact dup of the above
        _row("s3", "what is my order status"),
    ]
    kept, n_dropped = dedupe_exact(records)
    assert n_dropped == 1
    assert len(kept) == 2
    assert kept[0]["scenario_id"] == "s1"  # first occurrence kept


def test_dedupe_exact_does_not_cross_domains() -> None:
    records = [
        _row("s1", "cancel my order now", domain="retail"),
        _row("s1", "cancel my order now", domain="airline"),
    ]
    kept, n_dropped = dedupe_exact(records)
    assert n_dropped == 0
    assert len(kept) == 2


def test_dedupe_near_duplicates_merges_close_paraphrases_above_threshold() -> None:
    records = [
        _row("s1", "please cancel my pending order W1 right now immediately"),
        _row("s2", "please cancel my pending order W1 right now immediately please"),
        _row("s3", "what is the weather forecast for tomorrow in london"),
    ]
    kept, n_dropped = dedupe_near_duplicates(records, threshold=0.6, shingle_size=3)
    assert n_dropped == 1
    assert {r["scenario_id"] for r in kept} == {"s1", "s3"}


def test_dedupe_near_duplicates_threshold_is_configurable() -> None:
    records = [
        _row("s1", "please cancel my pending order W1 right now immediately"),
        _row("s2", "please cancel my pending order W1 right now immediately please"),
    ]
    # A very strict threshold treats the paraphrase as distinct.
    kept, n_dropped = dedupe_near_duplicates(records, threshold=0.999, shingle_size=3)
    assert n_dropped == 0
    assert len(kept) == 2


def test_dedupe_near_duplicates_buckets_by_domain_and_label_by_default() -> None:
    # Same text, but different label -> NOT deduped against each other
    # (label bucketing keeps them separate; each is a distinct data point).
    records = [
        _row("s1", "cancel my pending order right now", is_violation=True),
        _row("s2", "cancel my pending order right now", is_violation=False),
    ]
    kept, n_dropped = dedupe_near_duplicates(records, threshold=0.5)
    assert n_dropped == 0
    assert len(kept) == 2


_MULTILINE_SYSTEM_PROMPT = (
    "system: # Retail agent policy\n\n"
    "As a retail agent, you can help users:\n\n"
    "- cancel or modify pending orders\n"
    "- return or exchange delivered orders\n\n"
    "You must authenticate the user first.\n"
    "human: {human}"
)


def test_near_duplicate_comparison_text_strips_a_multiline_system_block() -> None:
    # render_prompt embeds the system message's OWN newlines verbatim (it
    # does not escape them), so only the FIRST physical line literally
    # starts with "system:" -- a naive per-line filter would leave the rest
    # of a multi-line policy block in place. This is the regression the
    # Part A stub smoke caught against real rendered retail/airline prompts.
    prompt = _MULTILINE_SYSTEM_PROMPT.format(human="cancel order W1")
    tail = near_duplicate_comparison_text(prompt)
    assert "Retail agent policy" not in tail
    assert "authenticate the user" not in tail
    assert tail == "human: cancel order W1"


def test_dedupe_near_duplicates_is_not_fooled_by_shared_multiline_boilerplate() -> None:
    # Two DIFFERENT scenarios sharing the identical multi-line system
    # preamble (as every decision in one domain does) must NOT collapse into
    # each other just because of that shared preamble.
    records = [
        _row("s1", _MULTILINE_SYSTEM_PROMPT.format(human="cancel order W1 immediately")),
        _row("s2", _MULTILINE_SYSTEM_PROMPT.format(human="look up my order status please")),
    ]
    kept, n_dropped = dedupe_near_duplicates(records, threshold=0.9, shingle_size=3)
    assert n_dropped == 0
    assert len(kept) == 2


def test_dedupe_runs_exact_then_near_and_reports_counts() -> None:
    records = [
        _row("s1", "cancel my pending order right now"),
        _row("s2", "cancel my pending order right now"),  # exact dup
        _row("s3", "cancel my pending order right now please"),  # near dup
        _row("s4", "what is the weather like today"),
    ]
    kept, report = dedupe(records, near_dup_threshold=0.6, shingle_size=3)
    assert report.n_input == 4
    assert report.n_exact_dropped == 1
    assert report.n_near_dropped == 1
    assert report.n_output == 2
    assert len(kept) == 2


# --- default_group_fields -------------------------------------------------


def test_default_group_fields_is_scenario_only_when_no_persona_present() -> None:
    records = [_row("s1", "x"), _row("s2", "y")]
    assert default_group_fields(records) == ("scenario_id",)


def test_default_group_fields_adds_persona_when_present() -> None:
    records = [_row("s1", "x", persona_id="ADV-0001"), _row("s2", "y")]
    assert default_group_fields(records) == ("scenario_id", "persona_id")


# --- stratified_group_split -----------------------------------------------


def _many_scenarios(
    n: int, *, variants: int = 4, domain: str = "retail"
) -> list[dict[str, object]]:
    records = []
    for i in range(n):
        for v in range(variants):
            # alternate label per scenario so both classes exist
            records.append(
                _row(
                    f"scn-{i}",
                    f"do the thing for scenario {i} variant {v}",
                    domain=domain,
                    is_violation=(i % 3 == 0),
                    variant_index=v,
                )
            )
    return records


def test_stratified_group_split_never_splits_a_scenario_across_splits() -> None:
    records = _many_scenarios(20)
    splits, report = stratified_group_split(records, val_frac=0.2, test_frac=0.2, seed=1)
    scenario_to_splits: dict[str, set[str]] = {}
    for split_name, rows in splits.items():
        for r in rows:
            scenario_to_splits.setdefault(str(r["scenario_id"]), set()).add(split_name)
    assert all(len(v) == 1 for v in scenario_to_splits.values())
    assert sum(report.counts[s]["n"] for s in report.counts) == len(records)


def test_stratified_group_split_is_deterministic_given_a_seed() -> None:
    records = _many_scenarios(15)
    splits_a, _ = stratified_group_split(records, seed=42)
    splits_b, _ = stratified_group_split(records, seed=42)
    assert [r["scenario_id"] for r in splits_a["train"]] == [
        r["scenario_id"] for r in splits_b["train"]
    ]
    assert [r["scenario_id"] for r in splits_a["test"]] == [
        r["scenario_id"] for r in splits_b["test"]
    ]


def test_stratified_group_split_different_seed_can_change_assignment() -> None:
    records = _many_scenarios(15)
    splits_a, _ = stratified_group_split(records, seed=1)
    splits_b, _ = stratified_group_split(records, seed=2)
    train_a = {r["scenario_id"] for r in splits_a["train"]}
    train_b = {r["scenario_id"] for r in splits_b["train"]}
    assert train_a != train_b  # not guaranteed by contract in general, but true for this fixture


def test_stratified_group_split_keeps_persona_id_grouped_when_present() -> None:
    # Two scenarios sharing a persona_id must land in the same split even
    # though they're different scenario_ids.
    records = [
        _row("s1", "enrol customer now", persona_id="ADV-0001", is_violation=True),
        _row("s2", "reject customer now", persona_id="ADV-0001", is_violation=False),
    ] + _many_scenarios(10, domain="advice-eligibility")
    splits, _ = stratified_group_split(records, val_frac=0.2, test_frac=0.2, seed=3)
    located = {
        name: {r["scenario_id"] for r in rows if r["persona_id"] == "ADV-0001"}
        for name, rows in splits.items()
    }
    nonempty = [v for v in located.values() if v]
    assert len(nonempty) == 1
    assert nonempty[0] == {"s1", "s2"}


def test_stratified_group_split_approximates_requested_fractions() -> None:
    records = _many_scenarios(60, variants=2)
    splits, report = stratified_group_split(records, val_frac=0.2, test_frac=0.2, seed=7)
    total = sum(report.counts[s]["n"] for s in report.counts)
    val_frac_actual = report.counts["val"]["n"] / total
    test_frac_actual = report.counts["test"]["n"] / total
    assert abs(val_frac_actual - 0.2) < 0.08
    assert abs(test_frac_actual - 0.2) < 0.08


def test_stratified_group_split_empty_input() -> None:
    splits, report = stratified_group_split([])
    assert splits == {"train": [], "val": [], "test": []}
    assert report.n_groups == 0
