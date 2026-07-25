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
