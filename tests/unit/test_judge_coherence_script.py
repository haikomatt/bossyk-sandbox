"""Hermetic tests for scripts/judge_coherence.py -- fake judge, no network."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from langchain_core.messages import AIMessage

SCRIPT_PATH = Path(__file__).parent.parent.parent / "scripts" / "judge_coherence.py"


def _import_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("judge_coherence_script", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass
class _ScriptedJudge:
    contents: list[str]
    calls: int = 0

    def invoke(self, _messages: list[Any]) -> AIMessage:
        c = self.contents[self.calls]
        self.calls += 1
        return AIMessage(content=c)


def test_imports_side_effect_free_and_defines_main() -> None:
    module = _import_script()
    assert callable(module.main)
    assert callable(module.judge_rows)


def test_judge_rows_labels_only_actioned_rows_in_place() -> None:
    module = _import_script()
    rows = [
        {"step_id": "t0", "prompt": "p0", "is_violation": True, "action": "cancel(W1)"},
        {"step_id": "t1", "prompt": "p1", "is_violation": False},  # no action -> skipped
        {"step_id": "t2", "prompt": "p2", "is_violation": True, "action": "gibberish"},
    ]
    judge = _ScriptedJudge(
        contents=['{"coherent": true, "reason": "ok"}', '{"coherent": false, "reason": "broken"}']
    )
    counts = module.judge_rows(rows, llm=judge, sleeper=lambda _s: None)

    assert counts == {"n_rows": 3, "n_judgeable": 2, "n_judged": 2, "n_error": 1, "n_unjudged": 0}
    assert rows[0]["is_error"] is False
    assert "is_error" not in rows[1]  # untouched: no action to judge
    assert rows[2]["is_error"] is True


def test_judge_rows_leaves_unjudged_rows_unlabelled() -> None:
    module = _import_script()
    rows = [{"step_id": "t0", "prompt": "p0", "is_violation": False, "action": "a"}]
    judge = _ScriptedJudge(contents=["not parseable"])  # -> None verdict
    counts = module.judge_rows(rows, llm=judge, sleeper=lambda _s: None)
    assert counts["n_judged"] == 0 and counts["n_unjudged"] == 1
    assert "is_error" not in rows[0]  # no fabricated label
