"""Hermetic tests for bossyk_sandbox.detector.train_backend -- the
Amendment-3 orchestration logic (calib slice, class weights, temperature
fit, manifest fields), exercised against StubTrainBackend so no torch/GPU is
needed. This is the brief's Phase A item 4 dry-run smoke, as unit tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from bossyk_sandbox.detector.train_backend import (
    StubTrainBackend,
    TrainConfig,
    class_weights,
    decision_text,
    run_training_cell,
)


def _rows(
    n_scenarios: int, per_scenario: int, *, violation_scenarios: set[int]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for s in range(n_scenarios):
        for i in range(per_scenario):
            rows.append(
                {
                    "row_id": f"s{s}:{i}",
                    "scenario_id": f"scenario-{s}",
                    "prompt": f"scenario {s} step {i}",
                    "action": "cancel" if s in violation_scenarios else "verify then cancel",
                    "is_violation": s in violation_scenarios,
                }
            )
    return rows


def test_decision_text_includes_prompt_and_action() -> None:
    text = decision_text({"prompt": "ctx", "action": "cancel(W1)"})
    assert "ctx" in text
    assert "cancel(W1)" in text


def test_class_weights_balanced_formula() -> None:
    y = [True] * 20 + [False] * 80
    w = class_weights(y)
    assert w[True] == 100 / (2 * 20)
    assert w[False] == 100 / (2 * 80)
    # minority class gets the bigger weight
    assert w[True] > w[False]


def test_class_weights_single_class_falls_back_to_one() -> None:
    assert class_weights([True, True, True]) == {True: 1.0, False: 1.0}
    assert class_weights([]) == {True: 1.0, False: 1.0}


def test_run_training_cell_produces_manifest_shaped_result(tmp_path: Path) -> None:
    train_rows = _rows(20, 10, violation_scenarios=set(range(0, 20, 3)))
    val_rows = _rows(5, 4, violation_scenarios={1})
    cfg = TrainConfig(
        family="deberta",
        train_domain="retail",
        seed=0,
        slice_seed=42,
        train_rows=train_rows,
        val_rows=val_rows,
        output_dir=tmp_path,
    )
    result = run_training_cell(cfg, StubTrainBackend())

    assert result.family == "deberta"
    assert result.train_domain == "retail"
    assert result.seed == 0
    assert result.slice_seed == 42
    assert result.checkpoint_dir.exists()
    assert result.n_train + result.n_calib == len(train_rows)
    assert result.n_val == len(val_rows)
    assert 0.0 <= result.train_violation_base_rate <= 1.0
    assert set(result.class_weights) == {"violation", "compliant"}
    assert result.calib_temperature > 0
    assert result.wall_seconds >= 0.0

    manifest = result.to_manifest_dict()
    for key in (
        "family",
        "train_domain",
        "seed",
        "slice_seed",
        "final_val_loss",
        "early_stop_epoch",
        "class_weights",
        "calib_temperature",
        "wall_seconds",
    ):
        assert key in manifest


def test_run_training_cell_calib_slice_is_disjoint_from_what_backend_sees_as_train(
    tmp_path: Path,
) -> None:
    train_rows = _rows(20, 10, violation_scenarios=set(range(0, 20, 2)))
    val_rows: list[dict[str, Any]] = []

    seen_train_ids: list[set[str]] = []

    class _RecordingBackend(StubTrainBackend):
        def train(self, *, train_rows: list[dict[str, Any]], **kwargs: Any) -> Any:
            seen_train_ids.append({r["row_id"] for r in train_rows})
            return super().train(train_rows=train_rows, **kwargs)

    cfg = TrainConfig(
        family="qwen_lora",
        train_domain="airline",
        seed=1,
        slice_seed=7,
        train_rows=train_rows,
        val_rows=val_rows,
        output_dir=tmp_path,
    )
    result = run_training_cell(cfg, _RecordingBackend())
    backend_train_ids = seen_train_ids[0]
    all_ids = {r["row_id"] for r in train_rows}
    calib_ids = all_ids - backend_train_ids
    assert len(calib_ids) == result.n_calib
    assert backend_train_ids.isdisjoint(calib_ids)


def test_run_training_cell_no_val_rows_still_produces_a_result(tmp_path: Path) -> None:
    train_rows = _rows(15, 8, violation_scenarios={0, 4, 8})
    cfg = TrainConfig(
        family="deberta",
        train_domain="advice-eligibility",
        seed=2,
        slice_seed=0,
        train_rows=train_rows,
        val_rows=[],
        output_dir=tmp_path,
    )
    result = run_training_cell(cfg, StubTrainBackend())
    assert result.n_val == 0
    assert result.checkpoint_dir.exists()


def test_run_training_cell_empty_calib_slice_falls_back_to_temperature_one(tmp_path: Path) -> None:
    # A single tiny scenario group can end up with all rows in train and
    # none in calib -- the orchestration must not crash or fabricate a fit.
    train_rows = _rows(1, 3, violation_scenarios={0})
    cfg = TrainConfig(
        family="deberta",
        train_domain="retail",
        seed=0,
        slice_seed=0,
        train_rows=train_rows,
        val_rows=[],
        output_dir=tmp_path,
    )
    result = run_training_cell(cfg, StubTrainBackend())
    if result.n_calib == 0:
        assert result.calib_temperature == 1.0


def test_stub_backend_score_is_deterministic() -> None:
    backend = StubTrainBackend()
    rows = _rows(3, 2, violation_scenarios={1})
    a = backend.score(family="deberta", checkpoint_dir=Path("/nonexistent"), rows=rows)
    b = backend.score(family="deberta", checkpoint_dir=Path("/nonexistent"), rows=rows)
    assert a == b


def test_stub_backend_score_separates_violation_from_compliant() -> None:
    backend = StubTrainBackend()
    rows = _rows(4, 20, violation_scenarios={0, 2})
    scores = backend.score(family="deberta", checkpoint_dir=Path("/nonexistent"), rows=rows)
    viol_scores = [s for s, r in zip(scores, rows, strict=True) if r["is_violation"]]
    compliant_scores = [s for s, r in zip(scores, rows, strict=True) if not r["is_violation"]]
    assert min(viol_scores) > max(compliant_scores)
