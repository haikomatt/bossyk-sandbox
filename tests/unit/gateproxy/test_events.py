"""Red-phase contract for the signed gate-event log.

One JSON object per line: ``{"event": {...}, "signature": {...}}``. The
signature covers ``canonicalize(event)`` with auditk's own canonical-JSON +
Ed25519 pattern (``auditk.attestation``), so a verifier needs only the
public key and this file -- same offline-verification story as the
evidence pack.
"""

from __future__ import annotations

import json
from pathlib import Path

from bossyk_sandbox.gateproxy.events import EventLog, verify_event_log


def _log(tmp_path: Path, signing_keys: tuple[Path, str]) -> EventLog:
    priv_path, _ = signing_keys
    return EventLog(
        path=tmp_path / "gate-events.jsonl",
        signer_key_path=priv_path,
        run_label="leg-b-nemotron",
    )


class TestAppend:
    def test_appends_one_signed_line_per_event(
        self, tmp_path: Path, signing_keys: tuple[Path, str]
    ) -> None:
        log = _log(tmp_path, signing_keys)
        log.append({"kind": "tool_call", "tool_name": "bash", "verdict": "allow"})
        log.append({"kind": "tool_call", "tool_name": "write", "verdict": "block"})
        lines = [json.loads(line) for line in log.path.read_text().splitlines()]
        assert len(lines) == 2
        assert all(set(entry) == {"event", "signature"} for entry in lines)

    def test_event_carries_run_label_and_timestamp(
        self, tmp_path: Path, signing_keys: tuple[Path, str]
    ) -> None:
        log = _log(tmp_path, signing_keys)
        log.append({"kind": "tool_call", "tool_name": "bash", "verdict": "allow"})
        entry = json.loads(log.path.read_text().splitlines()[0])
        assert entry["event"]["run_label"] == "leg-b-nemotron"
        assert entry["event"]["timestamp"].endswith("+00:00") or entry["event"][
            "timestamp"
        ].endswith("Z")

    def test_caller_fields_preserved_verbatim(
        self, tmp_path: Path, signing_keys: tuple[Path, str]
    ) -> None:
        log = _log(tmp_path, signing_keys)
        log.append(
            {
                "kind": "tool_call",
                "tool_name": "bash",
                "arguments": {"command": "curl http://x.example"},
                "verdict": "block",
                "policy_id": "no-network-egress",
                "reason": "network egress via curl",
            }
        )
        event = json.loads(log.path.read_text().splitlines()[0])["event"]
        assert event["policy_id"] == "no-network-egress"
        assert event["arguments"] == {"command": "curl http://x.example"}


class TestVerification:
    def test_intact_log_verifies(self, tmp_path: Path, signing_keys: tuple[Path, str]) -> None:
        _priv, pub_pem = signing_keys
        log = _log(tmp_path, signing_keys)
        for i in range(3):
            log.append({"kind": "tool_call", "tool_name": "bash", "verdict": "allow", "seq": i})
        result = verify_event_log(log.path, pub_pem)
        assert result.ok
        assert result.verified_count == 3
        assert result.bad_line_numbers == []

    def test_tampered_event_fails_verification_at_that_line(
        self, tmp_path: Path, signing_keys: tuple[Path, str]
    ) -> None:
        _priv, pub_pem = signing_keys
        log = _log(tmp_path, signing_keys)
        log.append({"kind": "tool_call", "tool_name": "bash", "verdict": "block", "seq": 0})
        log.append({"kind": "tool_call", "tool_name": "bash", "verdict": "allow", "seq": 1})
        lines = log.path.read_text().splitlines()
        doctored = json.loads(lines[0])
        doctored["event"]["verdict"] = "allow"  # rewrite history: block -> allow
        log.path.write_text("\n".join([json.dumps(doctored), lines[1]]) + "\n")
        result = verify_event_log(log.path, pub_pem)
        assert not result.ok
        assert result.bad_line_numbers == [1]
        assert result.verified_count == 1

    def test_wrong_public_key_fails_everything(
        self, tmp_path: Path, signing_keys: tuple[Path, str]
    ) -> None:
        from auditk.attestation.signer import generate_keypair

        log = _log(tmp_path, signing_keys)
        log.append({"kind": "tool_call", "tool_name": "bash", "verdict": "allow"})
        _other_priv, other_pub = generate_keypair(tmp_path / "other-key")
        result = verify_event_log(log.path, other_pub.read_text())
        assert not result.ok
        assert result.verified_count == 0
