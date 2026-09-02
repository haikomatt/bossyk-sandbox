"""Hermetic tests for scripts/detector_judge_baseline.py -- fake ChatModel,
no network, no real spend."""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from langchain_core.messages import AIMessage

SCRIPT_PATH = Path(__file__).parent.parent.parent / "scripts" / "detector_judge_baseline.py"


def _import() -> ModuleType:
    spec = importlib.util.spec_from_file_location("detector_judge_baseline_script", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    # JudgeItem is a @dataclass under `from __future__ import annotations` --
    # its string-annotation resolution needs the module registered in
    # sys.modules BEFORE exec_module (same gotcha as
    # test_generate_corpus_script.py / test_datagen_pipeline_smoke.py).
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


@dataclass
class _ScriptedJudge:
    contents: list[str | Exception]
    calls: int = 0

    def invoke(self, _messages: list[Any]) -> AIMessage:
        item = self.contents[self.calls]
        self.calls += 1
        if isinstance(item, Exception):
            raise item
        return AIMessage(content=item)


def _rows(n_pos: int, n_neg: int) -> list[dict[str, Any]]:
    rows = []
    for i in range(n_pos):
        rows.append(
            {"row_id": f"pos-{i}", "prompt": f"ctx{i}", "action": f"act{i}", "is_violation": True}
        )
    for i in range(n_neg):
        rows.append(
            {"row_id": f"neg-{i}", "prompt": f"ctx{i}", "action": f"act{i}", "is_violation": False}
        )
    return rows


def test_build_subsample_includes_all_positives_and_equal_negatives() -> None:
    m = _import()
    rows = _rows(n_pos=10, n_neg=100)
    items = m.build_subsample("retail", rows, seed=0)
    assert len(items) == 20
    assert sum(1 for it in items if it.is_violation) == 10
    assert sum(1 for it in items if not it.is_violation) == 10


def test_build_subsample_caps_at_available_negatives() -> None:
    m = _import()
    rows = _rows(n_pos=10, n_neg=3)
    items = m.build_subsample("retail", rows, seed=0)
    assert len(items) == 13  # 10 positives + only 3 negatives available


def test_build_subsample_deterministic_given_seed() -> None:
    m = _import()
    rows = _rows(n_pos=5, n_neg=50)
    a = [it.row_id for it in m.build_subsample("retail", rows, seed=7)]
    b = [it.row_id for it in m.build_subsample("retail", rows, seed=7)]
    assert a == b


def test_load_judge_checkpoint_round_trip(tmp_path: Path) -> None:
    m = _import()
    path = tmp_path / "ckpt.jsonl"
    path.write_text(
        json.dumps({"row_id": "a", "p_violation": 0.5})
        + "\n"
        + json.dumps({"row_id": "b", "p_violation": None})
        + "\n"
    )
    ckpt = m.load_judge_checkpoint(path)
    assert set(ckpt) == {"a", "b"}
    assert ckpt["a"]["p_violation"] == 0.5


def test_load_judge_checkpoint_missing_file_is_empty(tmp_path: Path) -> None:
    m = _import()
    assert m.load_judge_checkpoint(tmp_path / "nope.jsonl") == {}


def test_run_judge_items_writes_checkpoint_for_every_item(tmp_path: Path) -> None:
    m = _import()
    items = [
        m.JudgeItem(row_id="a", domain="retail", context="c0", action="ac0", is_violation=True),
        m.JudgeItem(row_id="b", domain="retail", context="c1", action="ac1", is_violation=False),
    ]
    llm = _ScriptedJudge(
        contents=[
            '{"p_violation": 0.9, "reason": "x"}',
            '{"p_violation": 0.1, "reason": "y"}',
        ]
    )
    cap = m.CapConfig(max_usd=100.0, max_calls=100)
    budget = m.BudgetGuard(cap)
    ckpt = tmp_path / "retail_judged.jsonl"
    counts = m.run_judge_items(
        items,
        policy_text="policy",
        llm=llm,
        budget=budget,
        checkpoint_path=ckpt,
        workers=2,
        sleeper=lambda _s: None,
    )
    assert counts["n_judged"] == 2
    written = m.load_judge_checkpoint(ckpt)
    assert set(written) == {"a", "b"}
    assert budget.calls == 2


def test_run_judge_items_skips_rows_already_checkpointed(tmp_path: Path) -> None:
    m = _import()
    ckpt = tmp_path / "retail_judged.jsonl"
    ckpt.write_text(
        json.dumps(
            {
                "row_id": "a",
                "domain": "retail",
                "is_violation": True,
                "p_violation": 0.7,
                "reason": "r",
            }
        )
        + "\n"
    )
    items = [
        m.JudgeItem(row_id="a", domain="retail", context="c0", action="ac0", is_violation=True),
        m.JudgeItem(row_id="b", domain="retail", context="c1", action="ac1", is_violation=False),
    ]
    llm = _ScriptedJudge(contents=['{"p_violation": 0.2, "reason": "y"}'])
    budget = m.BudgetGuard(m.CapConfig(max_usd=100.0, max_calls=100))
    counts = m.run_judge_items(
        items,
        policy_text="policy",
        llm=llm,
        budget=budget,
        checkpoint_path=ckpt,
        workers=1,
        sleeper=lambda _s: None,
    )
    assert counts["n_pending"] == 1  # "a" was already done
    assert llm.calls == 1
    written = m.load_judge_checkpoint(ckpt)
    assert written["a"]["p_violation"] == 0.7  # untouched


def test_run_judge_items_stops_new_dispatch_once_budget_cap_hit(tmp_path: Path) -> None:
    m = _import()
    items = [
        m.JudgeItem(
            row_id=f"i{i}", domain="retail", context="c", action="a", is_violation=bool(i % 2)
        )
        for i in range(10)
    ]
    llm = _ScriptedJudge(contents=['{"p_violation": 0.5, "reason": "x"}'] * 10)
    budget = m.BudgetGuard(m.CapConfig(max_usd=100.0, max_calls=3))  # backstop trips after 3 calls
    ckpt = tmp_path / "retail_judged.jsonl"
    m.run_judge_items(
        items,
        policy_text="policy",
        llm=llm,
        budget=budget,
        checkpoint_path=ckpt,
        workers=1,
        sleeper=lambda _s: None,
    )
    assert budget.calls <= 3  # never dispatches past the cap once it trips
    assert budget.cap_reason() == "calls"


def test_compute_domain_result_reports_auroc_fpr_fnr(tmp_path: Path) -> None:
    m = _import()
    items = [
        m.JudgeItem(row_id="p0", domain="retail", context="c", action="a", is_violation=True),
        m.JudgeItem(row_id="p1", domain="retail", context="c", action="a", is_violation=True),
        m.JudgeItem(row_id="n0", domain="retail", context="c", action="a", is_violation=False),
        m.JudgeItem(row_id="n1", domain="retail", context="c", action="a", is_violation=False),
    ]
    ckpt = tmp_path / "retail_judged.jsonl"
    with ckpt.open("w") as f:
        for row_id, p in [("p0", 0.9), ("p1", 0.4), ("n0", 0.1), ("n1", 0.6)]:
            f.write(json.dumps({"row_id": row_id, "p_violation": p}) + "\n")
    result = m.compute_domain_result(items, ckpt, threshold=0.5)
    assert result["n_scored"] == 4
    assert result["n_pos"] == 2
    # preds@0.5: p0=1(TP), p1=0(FN), n0=0(TN), n1=1(FP)
    assert result["tp"] == 1 and result["fn"] == 1 and result["fp"] == 1 and result["tn"] == 1
    assert result["fpr"] == 0.5
    assert result["fnr"] == 0.5


def test_compute_domain_result_excludes_unjudged_items(tmp_path: Path) -> None:
    m = _import()
    items = [
        m.JudgeItem(row_id="p0", domain="retail", context="c", action="a", is_violation=True),
        m.JudgeItem(row_id="p1", domain="retail", context="c", action="a", is_violation=True),
    ]
    ckpt = tmp_path / "retail_judged.jsonl"
    with ckpt.open("w") as f:
        f.write(json.dumps({"row_id": "p0", "p_violation": 0.8}) + "\n")
        f.write(json.dumps({"row_id": "p1", "p_violation": None}) + "\n")  # unjudged
    result = m.compute_domain_result(items, ckpt)
    assert result["n_scored"] == 1
    assert result["n_unjudged"] == 1


def test_prompt_hash_is_stable() -> None:
    m = _import()
    assert m.prompt_hash() == m.prompt_hash()
    assert len(m.prompt_hash()) == 64  # sha256 hex digest


def test_main_refuses_without_gate_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    m = _import()
    monkeypatch.delenv("RUN_DETECTOR_JUDGE_BASELINE", raising=False)
    assert m.main([]) == 1


def test_main_refuses_without_api_key(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    m = _import()
    monkeypatch.setenv("RUN_DETECTOR_JUDGE_BASELINE", "1")
    monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)
    monkeypatch.delenv("AGENT_API_KEY", raising=False)
    empty_env = tmp_path / "empty.env"
    empty_env.write_text("")
    assert m.main(["--env-file", str(empty_env)]) == 1


def test_main_env_file_default_points_at_sibling_bossyk_sandbox_checkout() -> None:
    m = _import()
    assert str(m.DEFAULT_ENV_PATH) == "/home/matt/Projects/bossyk-sandbox/.env"
