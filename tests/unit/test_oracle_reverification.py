"""Rung 4, slice 2: the bookend to the Part-A forgery demo.

Part A (`test_attestation_tcb_forgery.py`) showed a compromised tracer signs a
clean trace of a dirty run that `verify_signed_trace` accepts, indistinguishable
from the honest one -- because a signature proves integrity after signing, not
capture fidelity. This slice closes the loop for the *bindable* subset: with the
oracle observations bound in slice 1, an independent, reproducible oracle
re-checks them, and the same forgeries that PASS signature verification now FAIL
oracle re-verification.

Honest limits (see `probes/attestation/tcb_claim_boundary.md`) hold and are
tested: a check with no oracle, or one pinned to a different snapshot, is
reported `unbindable`, never a false contradiction.

Ground truth for the scenario: phone X is on the suppression list, so an honest
run BLOCKS the call and records `check_suppression -> {on_list: True}`.
"""

from __future__ import annotations

import tempfile
from datetime import UTC, datetime
from pathlib import Path

from auditk.attestation.signer import generate_keypair
from auditk.schema import Step, Trace

from bossyk_sandbox.evidence.oracle_binding import (
    OracleObservation,
    bind_oracle_observations,
)
from bossyk_sandbox.evidence.oracle_reverify import (
    StaticOracle,
    verify_against_oracle,
)
from bossyk_sandbox.evidence.signed_trace import (
    SignedTrace,
    build_signed_trace,
    verify_signed_trace,
)
from bossyk_sandbox.evidence.trace import build_trace, make_attested_step
from bossyk_sandbox.instruments.base import Decision, ProposedAction, Verdict

_EPOCH = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)
_PHONE = "07700900123"
_REF = "tps-suppression@2026-07-30#abc123"
_TRACE_ID = "outreach-rung4-bookend"
_CONFIG = "outreach-demo@sample"


def _benign_step() -> Step:
    return make_attested_step(
        _TRACE_ID,
        ProposedAction("lookup_prospect", {"phone": _PHONE}, "read the prospect record"),
        auto_decision=Decision(Verdict.ALLOW, "read-only lookup is permitted"),
        final_verdict=Verdict.ALLOW,
        step_id="s1-lookup",
        timestamp=_EPOCH,
    )


def _call_step(verdict: Verdict, claimed_on_list: bool) -> Step:
    """A place_call step bound to a suppression observation claiming
    `on_list=claimed_on_list`. The honest arm blocks with on_list=True; the
    relabel forgery allows with on_list=False."""
    step = make_attested_step(
        _TRACE_ID,
        ProposedAction("place_call", {"phone": _PHONE}, "cold-call the prospect"),
        auto_decision=Decision(Verdict.BLOCK, "check_suppression for the phone returned on-list"),
        final_verdict=verdict,
        step_id="s2-place-call",
        timestamp=_EPOCH.replace(second=1),
    )
    bind_oracle_observations(
        step,
        [
            OracleObservation(
                check_tool="check_suppression",
                key_arg="phone",
                key_value=_PHONE,
                claimed_result={"on_list": claimed_on_list},
                oracle_ref=_REF,
            )
        ],
    )
    return step


def _honest() -> Trace:
    # blocked call, faithfully recording on_list=True
    return build_trace(_TRACE_ID, _CONFIG, [_benign_step(), _call_step(Verdict.BLOCK, True)])


def _relabel_forgery() -> Trace:
    # the blocked call rewritten as ALLOW, its bound check flipped to on_list=False
    return build_trace(_TRACE_ID, _CONFIG, [_benign_step(), _call_step(Verdict.ALLOW, False)])


def _omit_forgery() -> Trace:
    # the blocked call dropped entirely -- the violation never happened
    return build_trace(_TRACE_ID, _CONFIG, [_benign_step()])


def _oracle() -> StaticOracle:
    # independent, reproducible: X IS suppressed, and the service logged that X
    # was queried (so an omitted step cannot hide the query).
    return StaticOracle(
        check_tool="check_suppression",
        ref=_REF,
        truth={_PHONE: {"on_list": True}},
        queried=frozenset({_PHONE}),
    )


def _sign(trace: Trace, key: Path) -> SignedTrace:
    return build_signed_trace(trace, signer_key_path=key)


# --- the signature cannot see the lie (Part-A restated) -----------------


def test_all_three_pass_signature_verification() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        priv, pub = generate_keypair(Path(tmp) / "k")
        pubkey = pub.read_text()
        for trace in (_honest(), _relabel_forgery(), _omit_forgery()):
            assert verify_signed_trace(_sign(trace, priv), pubkey) is True


# --- but the oracle can (the bookend) -----------------------------------


def test_honest_trace_passes_oracle_reverification() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        priv, _ = generate_keypair(Path(tmp) / "k")
        report = verify_against_oracle(_sign(_honest(), priv), [_oracle()])

    assert report.consistent is True
    assert [f.status for f in report.observations] == ["agree"]
    assert report.omissions == []


def test_relabel_forgery_fails_oracle_reverification_by_contradiction() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        priv, _ = generate_keypair(Path(tmp) / "k")
        report = verify_against_oracle(_sign(_relabel_forgery(), priv), [_oracle()])

    assert report.consistent is False
    contradiction = next(f for f in report.observations if f.status == "contradict")
    assert contradiction.claimed_result == {"on_list": False}
    assert contradiction.oracle_result == {"on_list": True}
    assert report.omissions == []


def test_omit_forgery_fails_oracle_reverification_by_omission() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        priv, _ = generate_keypair(Path(tmp) / "k")
        report = verify_against_oracle(_sign(_omit_forgery(), priv), [_oracle()])

    assert report.consistent is False
    # the oracle logged a query for X the trace never accounts for
    assert len(report.omissions) == 1
    assert report.omissions[0].key_value == _PHONE
    assert report.omissions[0].check_tool == "check_suppression"


# --- honest limits: unbindable, not false-positive ----------------------


def test_snapshot_ref_mismatch_is_unbindable_not_contradiction() -> None:
    stale = StaticOracle(
        check_tool="check_suppression",
        ref="tps-suppression@2020-01-01#old",  # a different snapshot than the bound ref
        truth={_PHONE: {"on_list": True}},
        queried=frozenset({_PHONE}),
    )
    with tempfile.TemporaryDirectory() as tmp:
        priv, _ = generate_keypair(Path(tmp) / "k")
        report = verify_against_oracle(_sign(_honest(), priv), [stale])

    assert [f.status for f in report.observations] == ["unbindable"]
    # unbindable is reported, but it is not a detected lie
    assert report.consistent is True


def test_no_oracle_for_check_is_unbindable() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        priv, _ = generate_keypair(Path(tmp) / "k")
        report = verify_against_oracle(_sign(_honest(), priv), [])

    assert [f.status for f in report.observations] == ["unbindable"]
    assert report.consistent is True
