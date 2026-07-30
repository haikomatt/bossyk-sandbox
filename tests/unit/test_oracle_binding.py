"""Rung 4, slice 1: binding oracle observations into the signed trace.

A `RequirePassedCheck` decision depends on the RESULT an earlier check
returned (`instruments/hardcoded_rule.py`), but that result never reaches the
signed trace: `make_step` records only the gate verdict and reason. This slice
surfaces the observation the decision relied on into the attested step, so it
is signed alongside everything else and a later pass (slice 2) can re-verify it
against an independent oracle.

Slice 1 is only the binding: the observation is derived, attached, signed, and
tamper-evident, and a trace with no oracle check is byte-identical to before.
Re-verification against a real oracle is slice 2.
"""

from __future__ import annotations

import tempfile
from datetime import UTC, datetime
from pathlib import Path

from auditk.attestation.signer import generate_keypair

from bossyk_sandbox.evidence.oracle_binding import (
    ORACLE_OBSERVATIONS_METADATA_KEY,
    OracleObservation,
    bind_oracle_observations,
    matching_checks,
    oracle_observation_for,
)
from bossyk_sandbox.evidence.sample import build_sample_trace
from bossyk_sandbox.evidence.signed_trace import (
    build_signed_trace,
    stable_trace_manifest,
    verify_signed_trace,
)
from bossyk_sandbox.evidence.trace import build_trace, make_attested_step
from bossyk_sandbox.instruments.base import Decision, ObservedAction, ProposedAction, Verdict

_EPOCH = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)
_PHONE = "07700900123"
_ORACLE_REF = "tps-suppression@2026-07-30#abc123"


def _gated_call() -> ProposedAction:
    return ProposedAction(
        tool_name="place_call",
        arguments={"phone": _PHONE},
        declared_intent="cold-call the prospect",
    )


def _passing_check() -> ObservedAction:
    return ObservedAction(
        ProposedAction(
            tool_name="check_suppression",
            arguments={"phone": _PHONE},
            declared_intent="check the suppression list before dialling",
        ),
        result={"on_list": False},
    )


def _bound_call_step() -> tuple[object, list[OracleObservation]]:
    """A single attested ALLOW step for the gated call, with the suppression
    check it relied on bound onto it."""
    gated = _gated_call()
    history = [_passing_check()]
    checks = matching_checks(gated, history, check_tool="check_suppression", key_arg="phone")
    observations = [
        oracle_observation_for(c, oracle_ref=_ORACLE_REF, key_arg="phone") for c in checks
    ]

    step = make_attested_step(
        "outreach-oracle-binding",
        gated,
        auto_decision=Decision(Verdict.ALLOW, "prior check_suppression for the phone passed"),
        final_verdict=Verdict.ALLOW,
        step_id="s1-place-call",
        timestamp=_EPOCH,
    )
    bind_oracle_observations(step, observations)
    return step, observations


# --- derivation ---------------------------------------------------------


def test_matching_checks_finds_the_observed_check_the_decision_relied_on() -> None:
    gated = _gated_call()
    checks = matching_checks(
        gated, [_passing_check()], check_tool="check_suppression", key_arg="phone"
    )
    assert len(checks) == 1
    assert checks[0].result == {"on_list": False}


def test_matching_checks_ignores_called_but_unobserved_and_mismatched_keys() -> None:
    gated = _gated_call()
    history: list[ProposedAction | ObservedAction] = [
        # called but never observed (a bare proposal, no result) -- not bindable
        ProposedAction("check_suppression", {"phone": _PHONE}),
        # observed, but for a different phone
        ObservedAction(
            ProposedAction("check_suppression", {"phone": "07700900999"}),
            result={"on_list": False},
        ),
    ]
    assert matching_checks(gated, history, check_tool="check_suppression", key_arg="phone") == []


# --- binding, signing, tamper-evidence ----------------------------------


def test_bound_observation_is_signed_and_verifies() -> None:
    step, observations = _bound_call_step()
    trace = build_trace("outreach-oracle-binding", "outreach-demo@sample", [step])

    with tempfile.TemporaryDirectory() as tmp:
        priv, pub = generate_keypair(Path(tmp) / "signer")
        signed = build_signed_trace(trace, signer_key_path=priv)

        assert verify_signed_trace(signed, pub.read_text()) is True

    # the observation is inside the signed payload, on the step it belongs to
    manifest = stable_trace_manifest(signed)
    step_meta = manifest["trace"]["steps"][0]["metadata"]
    bound = step_meta[ORACLE_OBSERVATIONS_METADATA_KEY]
    assert bound == [observations[0].model_dump(mode="json")]
    assert bound[0]["oracle_ref"] == _ORACLE_REF
    assert bound[0]["claimed_result"] == {"on_list": False}


def test_tampering_with_a_bound_observation_breaks_the_signature() -> None:
    step, _ = _bound_call_step()
    trace = build_trace("outreach-oracle-binding", "outreach-demo@sample", [step])

    with tempfile.TemporaryDirectory() as tmp:
        priv, pub = generate_keypair(Path(tmp) / "signer")
        signed = build_signed_trace(trace, signer_key_path=priv)

        # flip the claimed result AFTER signing: the signature must catch it
        signed.trace.steps[0].metadata[ORACLE_OBSERVATIONS_METADATA_KEY][0]["claimed_result"] = {
            "on_list": True
        }
        assert verify_signed_trace(signed, pub.read_text()) is False


# --- byte-identical for check-free traces -------------------------------


def test_check_free_trace_has_no_oracle_key_and_is_unchanged() -> None:
    # the sample trace binds nothing, so no step carries the oracle key and its
    # signed content is exactly what it was before this slice existed.
    sample = build_sample_trace()
    for step in sample.steps:
        assert ORACLE_OBSERVATIONS_METADATA_KEY not in step.metadata


def test_binding_an_empty_list_is_a_noop() -> None:
    step = make_attested_step(
        "outreach-oracle-binding",
        _gated_call(),
        auto_decision=Decision(Verdict.ALLOW, "allowed"),
        final_verdict=Verdict.ALLOW,
        step_id="s1",
        timestamp=_EPOCH,
    )
    before = dict(step.metadata)
    bind_oracle_observations(step, [])
    assert step.metadata == before
    assert ORACLE_OBSERVATIONS_METADATA_KEY not in step.metadata
