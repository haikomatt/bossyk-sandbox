"""Build the decisions dataset from a behavioral run (pure helpers).

Bridges the behavioral layer to the mechanistic one: pair each captured per-turn
decision prompt (`session.agent_prompts`) with its policy label (the turn's gate
verdict) into `DecisionItem`s, and serialise to the JSON `scripts/interp_capture.py`
loads. The live driving that produces `agent_prompts` + blocked-turn labels lives
in the producer script; these helpers are pure and torch-free.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from bossyk_sandbox.interp.capture_run import DecisionItem


def build_decisions(
    prompts: list[str],
    blocked_turns: set[int] | Iterable[int],
    *,
    actions: list[str] | None = None,
    errors: list[bool] | None = None,
) -> list[DecisionItem]:
    """Pair each turn's rendered prompt with its policy label. `is_violation` is
    membership in `blocked_turns` (the gate-blocked turn indices). `actions`, if
    given, must align 1:1 with `prompts` and supplies each turn's rendered action
    (the coherence judge's input). `errors`, if given, must align 1:1 and supplies
    the optional general-failure label. Raises if either length mismatches."""
    blocked = set(blocked_turns)
    if actions is not None and len(actions) != len(prompts):
        raise ValueError(f"actions length {len(actions)} != prompts length {len(prompts)}")
    if errors is not None and len(errors) != len(prompts):
        raise ValueError(f"errors length {len(errors)} != prompts length {len(prompts)}")
    return [
        DecisionItem(
            step_id=f"turn-{i}",
            prompt=prompt,
            is_violation=i in blocked,
            is_error=None if errors is None else errors[i],
            action=None if actions is None else actions[i],
        )
        for i, prompt in enumerate(prompts)
    ]


def decisions_to_json(items: list[DecisionItem]) -> list[dict[str, Any]]:
    """Serialise DecisionItems to the interp_capture items schema (is_error and
    action omitted when None, so `load_items` round-trips them back to None)."""
    rows: list[dict[str, Any]] = []
    for item in items:
        row: dict[str, Any] = {
            "step_id": item.step_id,
            "prompt": item.prompt,
            "is_violation": item.is_violation,
        }
        if item.is_error is not None:
            row["is_error"] = item.is_error
        if item.action is not None:
            row["action"] = item.action
        rows.append(row)
    return rows
