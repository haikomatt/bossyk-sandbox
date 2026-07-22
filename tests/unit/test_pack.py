from pathlib import Path

from auditk.attestation.signer import generate_keypair
from auditk.schema import Trace

from bossyk_sandbox.evidence.pack import build_and_sign, read_pack, verify_offline, write_pack
from bossyk_sandbox.evidence.trace import build_trace, make_step
from bossyk_sandbox.gate import Gate
from bossyk_sandbox.instruments.base import ProposedAction
from bossyk_sandbox.instruments.hardcoded_rule import RequireLookupBeforeCancel


def _sample_trace() -> Trace:
    gate = Gate(instruments=[RequireLookupBeforeCancel()])
    proposed = ProposedAction("get_reservation_details", {"reservation_id": "R1"})
    decision = gate.score(proposed)
    step = make_step(trace_id="t-1", proposed=proposed, decision=decision)
    return build_trace(trace_id="t-1", agent_config_ref="cfg-1", steps=[step])


def test_pack_build_sign_verify_roundtrip(tmp_path: Path) -> None:
    priv_path, pub_path = generate_keypair(tmp_path / "session")
    trace = _sample_trace()

    pack = build_and_sign(trace, agent_version="0.1.0", signer_key_path=priv_path)

    assert len(pack.signatures) == 1
    assert verify_offline(pack, pub_path.read_text()) is True


def test_verify_offline_rejects_tampered_pack(tmp_path: Path) -> None:
    priv_path, pub_path = generate_keypair(tmp_path / "session")
    trace = _sample_trace()
    pack = build_and_sign(trace, agent_version="0.1.0", signer_key_path=priv_path)

    pack.trace_summary.step_count = 999  # tamper after signing

    assert verify_offline(pack, pub_path.read_text()) is False


def test_pack_roundtrips_through_file(tmp_path: Path) -> None:
    priv_path, pub_path = generate_keypair(tmp_path / "session")
    trace = _sample_trace()
    pack = build_and_sign(trace, agent_version="0.1.0", signer_key_path=priv_path)

    pack_path = tmp_path / "pack.json"
    write_pack(pack, pack_path)
    loaded = read_pack(pack_path)

    assert verify_offline(loaded, pub_path.read_text()) is True
