"""Rung 4, slice 1: bind an oracle observation into an attested step.

A `RequirePassedCheck` (`instruments/hardcoded_rule.py`) allows a gated action
only when an earlier check returned a result its predicate accepted. That
result is the evidence behind the decision, yet `make_step`/`make_attested_step`
record only the gate verdict and reason, so the signed trace attests the
*decision* but not the *observed value it relied on*. This module surfaces that
observation onto the step, in the step metadata that is already part of the
signed payload, mirroring how the compliance control tags are carried
(`compliance/attribution.py`).

Binding alone buys nothing against a compromised tracer that fabricates the
observation too (see `probes/attestation/tcb_claim_boundary.md`, honest limit
#1): its value is realised only in slice 2, when an independent, reproducible
oracle re-checks the bound value. This slice is the binding and its
tamper-evidence; a trace that binds nothing is byte-identical to before.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from auditk.schema import Step
from pydantic import BaseModel

from bossyk_sandbox.instruments.base import ObservedAction, ProposedAction, _action_of

# Mirrors `attribution.CONTROLS_METADATA_KEY`: a namespaced key so the bound
# observations live alongside the control tags in the step's signed metadata
# without colliding with auditk's own fields.
ORACLE_OBSERVATIONS_METADATA_KEY = "bossyk_sandbox_oracle_observations"


class OracleObservation(BaseModel):
    """The result an external check returned, as the tracer recorded it, bound
    to the gated step whose decision relied on it.

    `claimed_result` is exactly what the tracer says the check returned; slice
    2 re-derives the true value from an independent oracle pinned by
    `oracle_ref` and flags a contradiction. `oracle_ref` identifies that
    reproducible source (e.g. a suppression-list version plus content hash),
    so re-verification is deterministic and offline."""

    check_tool: str
    key_arg: str
    key_value: str
    claimed_result: Any
    oracle_ref: str


def matching_checks(
    proposed: ProposedAction,
    history: Sequence[ProposedAction | ObservedAction],
    *,
    check_tool: str,
    key_arg: str,
) -> list[ObservedAction]:
    """The observed checks whose result a `RequirePassedCheck(gated_tool,
    check_tool, key_arg, ...)` decision on `proposed` could have relied on:
    same `check_tool`, same `key_arg` value, and actually observed (a bare
    proposal that was called but never observed is not bindable). Mirrors the
    rule's own matching so the bound observation is the one the gate saw."""
    key_value = proposed.arguments.get(key_arg)
    if not isinstance(key_value, str) or not key_value.strip():
        return []

    matches: list[ObservedAction] = []
    for item in history:
        action = _action_of(item)
        if action.tool_name != check_tool or action.arguments.get(key_arg) != key_value:
            continue
        if not isinstance(item, ObservedAction):
            continue  # called but never observed -- no result to bind
        matches.append(item)
    return matches


def oracle_observation_for(
    observed: ObservedAction, *, oracle_ref: str, key_arg: str
) -> OracleObservation:
    """Build the bindable record from an observed check. `key_value` is read
    off the check's own arguments, so it is the value that was actually
    checked."""
    return OracleObservation(
        check_tool=observed.action.tool_name,
        key_arg=key_arg,
        key_value=str(observed.action.arguments.get(key_arg)),
        claimed_result=observed.result,
        oracle_ref=oracle_ref,
    )


def bind_oracle_observations(step: Step, observations: Sequence[OracleObservation]) -> None:
    """Attach `observations` to `step` in its signed metadata. A no-op for an
    empty sequence, so steps whose decision relied on no oracle check keep
    byte-identical metadata and the key never appears. Mutates `step.metadata`
    in place, exactly as `make_attested_step` already does for the control
    tags."""
    if not observations:
        return
    step.metadata[ORACLE_OBSERVATIONS_METADATA_KEY] = [
        obs.model_dump(mode="json") for obs in observations
    ]
