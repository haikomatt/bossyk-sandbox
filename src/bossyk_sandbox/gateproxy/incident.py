"""Incident report builder (Build C). RED-phase skeleton: importable
names, no behaviour -- the implementation lands in Green."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from bossyk_sandbox.gateproxy.scorecard import Scorecard, ScorecardInputs


@dataclass(frozen=True)
class IncidentDecision:
    triggered: bool
    severity: str | None
    reason: str


@dataclass(frozen=True)
class ArtefactRow:
    filename: str
    sha256: str
    status: str


@dataclass(frozen=True)
class Incident:
    run_label: str
    task_name: str
    trace_id: str
    generated_at: datetime
    severity: str
    reason: str
    timeline: list[str]
    artefacts: list[ArtefactRow]
    pack_sha256: str | None
    gate_version: str | None


@dataclass(frozen=True)
class IncidentCliConfig:
    inputs: ScorecardInputs
    out_path: Path


def determine_incident(card: Scorecard) -> IncidentDecision:
    raise NotImplementedError


def build_incident(
    inputs: ScorecardInputs, card: Scorecard, decision: IncidentDecision
) -> Incident:
    raise NotImplementedError


def render_incident_html(incident: Incident) -> str:
    raise NotImplementedError


def build_incident_config(argv: list[str]) -> IncidentCliConfig:
    raise NotImplementedError


def main(argv: list[str] | None = None) -> None:
    raise NotImplementedError
