"""Rung 4, slice 2: re-verify a signed trace against an independent oracle.

`verify_signed_trace` answers "was this altered since signing". This module
answers the distinct, harder question "does this agree with reality" -- and the
two are kept as separate functions on purpose, so the claim boundary stays
legible (`probes/attestation/tcb_claim_boundary.md`). A signature cannot see a
compromised tracer's lie; an independent, reproducible oracle can, for the
bindable subset.

Two forgery modes, two mechanisms:

* **relabel** -- the tracer flips a check's recorded result. The bound
  `claimed_result` contradicts the oracle's true value. Caught as a
  `contradict` finding.
* **omit** -- the tracer drops the violating step. The oracle independently
  logged that its check was queried for that key, yet no bound observation in
  the trace accounts for it. Caught as an omission finding.

Honest limits are enforced, not hidden: a check with no oracle, or a bound
observation pinned to a different snapshot than the oracle holds, is reported
`unbindable` -- surfaced, never a false contradiction. Binding to a
same-key-signed payload buys nothing on its own (honest limit #1); the value is
this independent re-check.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from pydantic import BaseModel

from bossyk_sandbox.evidence.oracle_binding import (
    ORACLE_OBSERVATIONS_METADATA_KEY,
    OracleObservation,
)
from bossyk_sandbox.evidence.signed_trace import SignedTrace

# Re-verification statuses for a single bound observation.
AGREE = "agree"
CONTRADICT = "contradict"
UNBINDABLE = "unbindable"


class OracleSource(Protocol):
    """An independent, reproducible source of truth for one check.

    `ref` pins the snapshot it answers for; re-verification only trusts it
    against a bound observation carrying the same `oracle_ref`, so the check is
    deterministic and offline. `evaluate` returns the value the check WOULD
    return under that snapshot; `observed_keys` is the set of keys the source
    independently logged as queried, used to catch omitted steps.

    `check_tool`/`ref` are read-only (properties) so a frozen implementation
    like `StaticOracle` satisfies the protocol."""

    @property
    def check_tool(self) -> str: ...

    @property
    def ref(self) -> str: ...

    def evaluate(self, key_value: str) -> Any: ...

    def observed_keys(self) -> set[str]: ...


@dataclass(frozen=True)
class StaticOracle:
    """A reproducible in-memory oracle backed by a pinned snapshot: a mapping
    of key to true result, plus the set of keys the snapshot logged as queried.
    Deterministic and offline -- the reference `OracleSource` for a
    suppression list, a clock reading, or an eligibility source."""

    check_tool: str
    ref: str
    truth: Mapping[str, Any]
    queried: frozenset[str] = field(default_factory=frozenset)

    def evaluate(self, key_value: str) -> Any:
        return self.truth.get(key_value)

    def observed_keys(self) -> set[str]:
        return set(self.queried)


class ObservationFinding(BaseModel):
    """The verdict of re-checking one bound observation against its oracle."""

    step_id: str
    check_tool: str
    key_value: str
    status: str
    claimed_result: Any
    oracle_result: Any = None
    detail: str = ""


class OmissionFinding(BaseModel):
    """A key the oracle independently logged as queried that no bound
    observation in the trace accounts for -- the fingerprint of a dropped
    step."""

    check_tool: str
    key_value: str
    oracle_ref: str


class OracleReport(BaseModel):
    observations: list[ObservationFinding]
    omissions: list[OmissionFinding]

    @property
    def contradictions(self) -> list[ObservationFinding]:
        return [f for f in self.observations if f.status == CONTRADICT]

    @property
    def unbindable(self) -> list[ObservationFinding]:
        return [f for f in self.observations if f.status == UNBINDABLE]

    @property
    def consistent(self) -> bool:
        """True when nothing contradicts reality and nothing was omitted. An
        `unbindable` observation is NOT a lie -- it was simply not re-checkable
        -- so it is surfaced but does not flip consistency."""
        return not self.contradictions and not self.omissions


def verify_against_oracle(signed: SignedTrace, oracles: Sequence[OracleSource]) -> OracleReport:
    """Re-check every bound observation in `signed` against the matching
    oracle, and reconcile each oracle's independently-logged query keys against
    what the trace accounts for. See the module docstring for the two mechanisms
    and the honest limits."""
    by_tool: dict[str, OracleSource] = {oracle.check_tool: oracle for oracle in oracles}
    observations: list[ObservationFinding] = []
    # check_tool -> the key_values the trace carries a bound observation for.
    accounted: dict[str, set[str]] = {}

    for step in signed.trace.steps:
        for raw in step.metadata.get(ORACLE_OBSERVATIONS_METADATA_KEY, []):
            obs = OracleObservation.model_validate(raw)
            accounted.setdefault(obs.check_tool, set()).add(obs.key_value)
            observations.append(_check_observation(step.step_id, obs, by_tool.get(obs.check_tool)))

    omissions: list[OmissionFinding] = []
    for oracle in oracles:
        present = accounted.get(oracle.check_tool, set())
        for key in sorted(oracle.observed_keys() - present):
            omissions.append(
                OmissionFinding(check_tool=oracle.check_tool, key_value=key, oracle_ref=oracle.ref)
            )

    return OracleReport(observations=observations, omissions=omissions)


def _check_observation(
    step_id: str, obs: OracleObservation, oracle: OracleSource | None
) -> ObservationFinding:
    if oracle is None:
        return ObservationFinding(
            step_id=step_id,
            check_tool=obs.check_tool,
            key_value=obs.key_value,
            status=UNBINDABLE,
            claimed_result=obs.claimed_result,
            detail=f"no oracle for check_tool {obs.check_tool!r}",
        )
    if oracle.ref != obs.oracle_ref:
        return ObservationFinding(
            step_id=step_id,
            check_tool=obs.check_tool,
            key_value=obs.key_value,
            status=UNBINDABLE,
            claimed_result=obs.claimed_result,
            detail=f"oracle snapshot {oracle.ref!r} != bound {obs.oracle_ref!r}",
        )
    true_result = oracle.evaluate(obs.key_value)
    status = AGREE if true_result == obs.claimed_result else CONTRADICT
    return ObservationFinding(
        step_id=step_id,
        check_tool=obs.check_tool,
        key_value=obs.key_value,
        status=status,
        claimed_result=obs.claimed_result,
        oracle_result=true_result,
    )
