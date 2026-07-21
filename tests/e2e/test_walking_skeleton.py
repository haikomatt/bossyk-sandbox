import os
from pathlib import Path

import pytest
from auditk.attestation.signer import generate_keypair

from bossyk_sandbox.evidence.pack import build_and_sign, verify_offline
from bossyk_sandbox.gate import Gate
from bossyk_sandbox.instruments.base import Verdict
from bossyk_sandbox.instruments.hardcoded_rule import RequireLookupBeforeCancel
from bossyk_sandbox.runtime.stub_agent import run_stub_session

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_SANDBOX_E2E") != "1",
    reason="Set RUN_SANDBOX_E2E=1 to run the walking-skeleton E2E test.",
)


def test_walking_skeleton_stub_agent_end_to_end(tmp_path: Path) -> None:
    gate = Gate(instruments=[RequireLookupBeforeCancel()])
    result = run_stub_session(gate)

    decisions_by_key = {
        f"{call.tool_name}:{call.arguments.get('reservation_id')}": decision
        for call, decision in result.decisions
    }

    benign_decision = decisions_by_key["cancel_reservation:RES-BENIGN"]
    unauthorised_decision = decisions_by_key["cancel_reservation:RES-UNAUTHORISED"]

    # (a) benign action allowed + appears in the trace
    assert benign_decision.verdict is Verdict.ALLOW
    assert any(
        step.action.payload["tool_name"] == "cancel_reservation"
        and step.action.payload["arguments"].get("reservation_id") == "RES-BENIGN"
        for step in result.steps
    )

    # (b) planted unauthorised action blocked before execution
    assert unauthorised_decision.verdict is Verdict.BLOCK
    assert not any(
        call.tool_name == "cancel_reservation"
        and call.arguments.get("reservation_id") == "RES-UNAUTHORISED"
        for call in result.executed
    )

    # (c) signed EvidencePack produced and verifies offline
    priv_path, pub_path = generate_keypair(tmp_path / "session")
    pack = build_and_sign(result.trace, agent_version="0.1.0", signer_key_path=priv_path)

    assert verify_offline(pack, pub_path.read_text()) is True
