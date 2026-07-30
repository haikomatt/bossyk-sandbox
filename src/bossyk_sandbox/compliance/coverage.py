"""Rolls the per-step control tags up to the whole trace: which compliance
controls the evidence discharges, and which steps evidence each one. This
is the pack-level answer an auditor reads -- "this record covers Art. 12,
CC7.3, ... and here are the actions behind each" -- computed purely from
the tags `make_attested_step` wrote into each step's metadata.
"""

from __future__ import annotations

from auditk.schema import Trace

from bossyk_sandbox.compliance.attribution import CONTROLS_METADATA_KEY


def trace_control_coverage(trace: Trace) -> dict[str, list[str]]:
    """Maps each control ref discharged anywhere in the trace to the ids of
    the steps that discharge it, both in trace order (deterministic, so the
    roll-up is byte-stable). A step with no control tags contributes
    nothing; an untagged trace rolls up to an empty mapping."""
    coverage: dict[str, list[str]] = {}
    for step in trace.steps:
        for tag in step.metadata.get(CONTROLS_METADATA_KEY, []):
            coverage.setdefault(tag["ref"], []).append(step.step_id)
    return coverage


def trace_deterministic_control_refs(trace: Trace) -> set[str]:
    """The control refs discharged by at least one DETERMINISTIC-basis tag
    somewhere in the trace (bossyk-sandbox slice 2, P7) -- the subset for
    which the evidence pack can drop the "directional mapping" disclaimer,
    because a reproducible check (not a judged/structural inference) backs
    the discharge. Additive alongside `trace_control_coverage`, which stays
    untouched: reads the same per-step `CONTROLS_METADATA_KEY` metadata
    `make_attested_step` already writes -- each serialized `ControlTag`
    dict carries a `discharge` key once `ControlTag.discharge` exists, via
    `tag.model_dump()`."""
    deterministic: set[str] = set()
    for step in trace.steps:
        for tag in step.metadata.get(CONTROLS_METADATA_KEY, []):
            if tag.get("discharge") == "deterministic":
                deterministic.add(tag["ref"])
    return deterministic
