"""Preset adversarial replays for the console's turn-by-turn playback view.

A replay preset is a committed, deterministic reconstruction of a live
gate-save: an ordered list of proposed tool calls driven through the *real*
domain fast-rule gate, reproducing the verdict sequence and harm delta a
committed live-run artifact reports. It is NOT a replay of a live LLM
transcript (the bench runs do not persist per-turn transcripts) -- the
load-bearing quantities are the gate verdicts and the harm delta, both
recomputed here from the same gate the live run used; the argument strings are
cosmetic placeholders. Each crossing turn carries the `probe_id` of the
committed-artifact row it stands for, so a reviewer (or the grounding test) can
check the reconstruction against the measured run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from auditk.schema import Step, Trace

from bossyk_sandbox.domains import domain_config
from bossyk_sandbox.evidence.trace import build_trace, make_attested_step
from bossyk_sandbox.gate import Gate
from bossyk_sandbox.instruments.base import ProposedAction, Verdict

# Repo root: this file is src/bossyk_sandbox/console/replay.py.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_PRESETS_DIR = _REPO_ROOT / "story" / "replay"


@dataclass(frozen=True)
class ReplayTurn:
    """One proposed tool call in a preset. `role` is "baseline" (demo scaffold
    -- a compliant lookup+cancel pair) or "crossing" (a measured gate-save,
    identified by `probe_id`)."""

    proposed: ProposedAction
    label: str
    role: str
    probe_id: str | None = None


@dataclass(frozen=True)
class ReplayPreset:
    preset_id: str
    title: str
    domain: str
    source_artifact: str
    story_claim: str
    turns: list[ReplayTurn]
    expected_verdicts: list[str]
    expected_harm_prevented: int
    expected_harm_delta: int


@dataclass(frozen=True)
class ReplayResult:
    """The deterministic outcome of driving a preset through the domain gate."""

    verdicts: list[str]
    steps: list[Step]
    harm_prevented: int
    harm_delta: int
    trace: Trace


def load_replay_preset(preset_id: str) -> ReplayPreset:
    """Load a committed replay preset from `story/replay/<preset_id>.json`.

    Raises `FileNotFoundError` for an unknown preset and `KeyError` for an
    unregistered domain (via `domain_config` at drive time), rather than
    silently substituting a default.
    """
    data = json.loads((_PRESETS_DIR / f"{preset_id}.json").read_text())
    turns = [
        ReplayTurn(
            proposed=ProposedAction(turn["tool_name"], turn["arguments"]),
            label=turn["label"],
            role=turn["role"],
            probe_id=turn.get("probe_id"),
        )
        for turn in data["turns"]
    ]
    expected = data["expected"]
    return ReplayPreset(
        preset_id=data["preset_id"],
        title=data["title"],
        domain=data["domain"],
        source_artifact=data["source_artifact"],
        story_claim=data["story_claim"],
        turns=turns,
        expected_verdicts=list(expected["verdicts"]),
        expected_harm_prevented=int(expected["harm_prevented"]),
        expected_harm_delta=int(expected["harm_delta"]),
    )


def build_gate(preset: ReplayPreset) -> Gate:
    """A fresh gate wired with the preset domain's fast rules -- the same rules
    the live run gated against."""
    return Gate(instruments=domain_config(preset.domain).fast_rules_factory())


def trace_id_for(preset: ReplayPreset) -> str:
    return f"replay-{preset.preset_id}"


def drive_replay(preset: ReplayPreset) -> ReplayResult:
    """Drive every turn through a fresh domain gate, attesting each as a step.

    Pure and deterministic (no network, no clock-dependent output beyond the
    step timestamps): the verdict sequence and harm delta recompute from the
    gate alone. `harm_prevented`/`harm_delta` count crossing turns the gate
    blocked -- the un-interrupted counterfactual is one harm unit per crossing,
    reduced to zero for each block.
    """
    gate = build_gate(preset)
    verdicts: list[str] = []
    steps: list[Step] = []
    harm_prevented = 0

    for turn in preset.turns:
        decision = gate.score(turn.proposed)
        if decision.verdict is Verdict.ALLOW:
            gate.record(turn.proposed)
        steps.append(
            make_attested_step(
                trace_id=trace_id_for(preset),
                proposed=turn.proposed,
                auto_decision=decision,
                final_verdict=decision.verdict,
            )
        )
        verdicts.append(decision.verdict.value)
        if turn.role == "crossing" and decision.verdict is Verdict.BLOCK:
            harm_prevented += 1

    trace = build_trace(
        trace_id=trace_id_for(preset),
        agent_config_ref=f"replay:{preset.preset_id}",
        steps=steps,
    )
    return ReplayResult(
        verdicts=verdicts,
        steps=steps,
        harm_prevented=harm_prevented,
        harm_delta=harm_prevented,
        trace=trace,
    )
