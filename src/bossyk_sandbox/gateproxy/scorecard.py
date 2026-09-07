"""RED-phase skeleton -- see tests/unit/gateproxy/test_scorecard.py."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ScorecardInputs:
    run_label: str
    task_name: str
    trace_path: Path
    evidence_pack_path: Path
    gate_events_path: Path
    report_md_path: Path | None
    pack_public_key_pem: str
    gate_public_key_pem: str
    generated_at: datetime


def build_scorecard(inputs: ScorecardInputs) -> Any:
    raise NotImplementedError("scorecard Green phase pending")


def render_scorecard_html(scorecard: Any) -> str:
    raise NotImplementedError("scorecard Green phase pending")


def build_scorecard_config(argv: list[str]) -> Any:
    raise NotImplementedError("scorecard Green phase pending")


def main(argv: list[str] | None = None) -> None:
    raise NotImplementedError("scorecard Green phase pending")
