"""Red-phase contract for the incident report builder (Build C).

No LLM anywhere: this is deterministic templating over the same three
artefacts the scorecard reads (trace, evidence pack, gate events) plus
their public keys. An incident exists only when at least one of three
conditions holds: any gate BLOCK event, a corroboration failure (a BLOCK
with no matching refusal in the trace), or a verification failure (pack
or gate-log signature does not check out). When none holds, nothing is
written -- the CLI prints "no incident: <reason>" and exits 0.

`build_incident` reuses `bossyk_sandbox.gateproxy.scorecard.build_scorecard`
for the gate/corroboration/verification/provenance numbers (extend, don't
duplicate) and does its own light parsing of the raw events + trace only
for what the scorecard doesn't already expose: per-event timestamps and
step ordering, needed for the prose timeline.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from auditk.attestation.canonical import canonicalize
from auditk.attestation.signer import LocalEd25519Signer

from bossyk_sandbox.gateproxy.events import EventLog
from bossyk_sandbox.gateproxy.incident import (
    build_incident_config,
    determine_incident,
    main,
    render_incident_html,
)
from bossyk_sandbox.gateproxy.scorecard import ScorecardInputs, build_scorecard

_GENERATED_AT = datetime(2026, 9, 7, 23, 0, 0, tzinfo=UTC)

_REFUSAL_TEXT = (
    "[bossyk gate] BLOCKED: this response proposed actions that violate policy.\n"
    "- no-credential-reads: access to credential path matching '.env'"
)


def _write_trace(path: Path, steps: list[dict[str, Any]]) -> None:
    path.write_text(
        json.dumps({"trace_id": "sess-incident-1", "source_adapter": "pi", "steps": steps})
    )


def _write_pack(path: Path, signer_key: Path, *, drift_score: float = 0.0) -> None:
    manifest = {
        "pack_id": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        "spec_version": "v0.1",
        "issued_at": "2026-09-07T22:30:00+00:00",
        "drift_metrics": {"drift_score": drift_score, "flagged_steps": [], "per_step": {}},
    }
    signature = LocalEd25519Signer(signer_key).sign(canonicalize(manifest))
    path.write_text(json.dumps({**manifest, "signatures": [signature.model_dump(mode="json")]}))


def _write_events(path: Path, signer_key: Path, events: list[dict[str, Any]]) -> None:
    log = EventLog(path=path, signer_key_path=signer_key, run_label="leg-b-incident-test")
    for event in events:
        log.append(event)


_ALLOW_EVENT = {
    "kind": "tool_call",
    "tool_name": "bash",
    "arguments": {"command": "ls"},
    "verdict": "allow",
    "policy_id": None,
    "reason": "no instrument blocked",
    "model": "test-model",
}

_BLOCK_EVENT = {
    "kind": "tool_call",
    "tool_name": "read",
    "arguments": {"path": ".env"},
    "verdict": "block",
    "policy_id": "no-credential-reads",
    "reason": "no-credential-reads: access to credential path matching '.env'",
    "model": "test-model",
}

_CLEAN_STEPS = [
    {
        "step_id": "s1",
        "actor": "user",
        "timestamp": "2026-09-07T20:00:00+00:00",
        "action": {"type": "utterance", "payload": {"text": "build logsum"}},
    },
    {
        "step_id": "s2",
        "actor": "agent",
        "timestamp": "2026-09-07T20:00:01+00:00",
        "action": {"type": "tool_call", "payload": {"name": "bash", "input": {"command": "ls"}}},
    },
]

_BLOCK_STEPS_WITH_REFUSAL = [
    *_CLEAN_STEPS,
    {
        "step_id": "cc63f000",
        "actor": "agent",
        "timestamp": "2026-09-07T20:00:05+00:00",
        "action": {"type": "utterance", "payload": {"text": _REFUSAL_TEXT}},
    },
]


@pytest.fixture()
def signer(signing_keys: tuple[Path, str]) -> Path:
    priv, _ = signing_keys
    return priv


@pytest.fixture()
def pub_pem(signing_keys: tuple[Path, str]) -> str:
    _, pub = signing_keys
    return pub


def _inputs(
    tmp_path: Path,
    signer: Path,
    pub_pem: str,
    *,
    steps: list[dict[str, Any]],
    events: list[dict[str, Any]],
    tamper_pack: bool = False,
    tamper_gate_log: bool = False,
) -> ScorecardInputs:
    trace_path = tmp_path / "trace.json"
    pack_path = tmp_path / "evidence-pack.json"
    events_path = tmp_path / "gate-events.jsonl"
    _write_trace(trace_path, steps)
    _write_pack(pack_path, signer)
    _write_events(events_path, signer, events)
    if tamper_pack:
        doctored = json.loads(pack_path.read_text())
        doctored["drift_metrics"]["drift_score"] = 999.0
        pack_path.write_text(json.dumps(doctored))
    if tamper_gate_log:
        lines = events_path.read_text().splitlines()
        entry = json.loads(lines[0])
        entry["event"]["verdict"] = "block" if entry["event"]["verdict"] == "allow" else "allow"
        events_path.write_text("\n".join([json.dumps(entry), *lines[1:]]) + "\n")
    return ScorecardInputs(
        run_label="leg-b-incident-test",
        task_name="logsum-hostile (incident test)",
        trace_path=trace_path,
        evidence_pack_path=pack_path,
        gate_events_path=events_path,
        report_md_path=None,
        pack_public_key_pem=pub_pem,
        gate_public_key_pem=pub_pem,
        generated_at=_GENERATED_AT,
    )


class TestDetermineIncident:
    def test_no_block_no_incident(self, tmp_path: Path, signer: Path, pub_pem: str) -> None:
        inputs = _inputs(tmp_path, signer, pub_pem, steps=_CLEAN_STEPS, events=[_ALLOW_EVENT])
        card = build_scorecard(inputs)
        decision = determine_incident(card)
        assert decision.triggered is False
        assert decision.severity is None
        assert decision.reason

    def test_block_with_corroboration_is_contained(
        self, tmp_path: Path, signer: Path, pub_pem: str
    ) -> None:
        inputs = _inputs(
            tmp_path,
            signer,
            pub_pem,
            steps=_BLOCK_STEPS_WITH_REFUSAL,
            events=[_ALLOW_EVENT, _BLOCK_EVENT],
        )
        card = build_scorecard(inputs)
        decision = determine_incident(card)
        assert decision.triggered is True
        assert decision.severity == "contained"

    def test_block_without_corroboration_is_integrity(
        self, tmp_path: Path, signer: Path, pub_pem: str
    ) -> None:
        inputs = _inputs(
            tmp_path, signer, pub_pem, steps=_CLEAN_STEPS, events=[_ALLOW_EVENT, _BLOCK_EVENT]
        )
        card = build_scorecard(inputs)
        decision = determine_incident(card)
        assert decision.triggered is True
        assert decision.severity == "integrity"

    def test_verification_failure_is_integrity_even_with_no_block(
        self, tmp_path: Path, signer: Path, pub_pem: str
    ) -> None:
        inputs = _inputs(
            tmp_path,
            signer,
            pub_pem,
            steps=_CLEAN_STEPS,
            events=[_ALLOW_EVENT],
            tamper_pack=True,
        )
        card = build_scorecard(inputs)
        decision = determine_incident(card)
        assert decision.triggered is True
        assert decision.severity == "integrity"

    def test_tampered_gate_log_is_integrity(
        self, tmp_path: Path, signer: Path, pub_pem: str
    ) -> None:
        inputs = _inputs(
            tmp_path,
            signer,
            pub_pem,
            steps=_CLEAN_STEPS,
            events=[_ALLOW_EVENT],
            tamper_gate_log=True,
        )
        card = build_scorecard(inputs)
        decision = determine_incident(card)
        assert decision.triggered is True
        assert decision.severity == "integrity"


class TestRenderContained:
    def test_severity_line_and_timeline_reference_ids(
        self, tmp_path: Path, signer: Path, pub_pem: str
    ) -> None:
        inputs = _inputs(
            tmp_path,
            signer,
            pub_pem,
            steps=_BLOCK_STEPS_WITH_REFUSAL,
            events=[_ALLOW_EVENT, _BLOCK_EVENT],
        )
        from bossyk_sandbox.gateproxy.incident import build_incident

        card = build_scorecard(inputs)
        decision = determine_incident(card)
        incident = build_incident(inputs, card, decision)
        html = render_incident_html(incident)
        assert "contained" in html.lower()
        assert "no-credential-reads" in html
        assert "cc63f000" in html
        assert "read" in html
        assert ".env" in html

    def test_artefact_inventory_present(self, tmp_path: Path, signer: Path, pub_pem: str) -> None:
        from bossyk_sandbox.gateproxy.incident import build_incident

        inputs = _inputs(
            tmp_path,
            signer,
            pub_pem,
            steps=_BLOCK_STEPS_WITH_REFUSAL,
            events=[_ALLOW_EVENT, _BLOCK_EVENT],
        )
        card = build_scorecard(inputs)
        incident = build_incident(inputs, card, determine_incident(card))
        html = render_incident_html(incident)
        for filename in ("trace.json", "evidence-pack.json", "gate-events.jsonl"):
            assert filename in html

    def test_footer_states_not_legal_advice(
        self, tmp_path: Path, signer: Path, pub_pem: str
    ) -> None:
        from bossyk_sandbox.gateproxy.incident import build_incident

        inputs = _inputs(
            tmp_path,
            signer,
            pub_pem,
            steps=_BLOCK_STEPS_WITH_REFUSAL,
            events=[_ALLOW_EVENT, _BLOCK_EVENT],
        )
        card = build_scorecard(inputs)
        incident = build_incident(inputs, card, determine_incident(card))
        html = render_incident_html(incident)
        lower = html.lower()
        assert "not legal advice" in lower
        assert "73" in html  # EU AI Act Art 73 reference

    def test_self_contained_and_script_free(
        self, tmp_path: Path, signer: Path, pub_pem: str
    ) -> None:
        from bossyk_sandbox.gateproxy.incident import build_incident

        inputs = _inputs(
            tmp_path,
            signer,
            pub_pem,
            steps=_BLOCK_STEPS_WITH_REFUSAL,
            events=[_ALLOW_EVENT, _BLOCK_EVENT],
        )
        card = build_scorecard(inputs)
        incident = build_incident(inputs, card, determine_incident(card))
        html = render_incident_html(incident)
        assert "<script" not in html.lower()

    def test_untrusted_argument_text_is_escaped(
        self, tmp_path: Path, signer: Path, pub_pem: str
    ) -> None:
        from bossyk_sandbox.gateproxy.incident import build_incident

        hostile_event = {
            **_BLOCK_EVENT,
            "arguments": {"path": "<script>alert('x')</script>"},
        }
        inputs = _inputs(
            tmp_path,
            signer,
            pub_pem,
            steps=_BLOCK_STEPS_WITH_REFUSAL,
            events=[_ALLOW_EVENT, hostile_event],
        )
        card = build_scorecard(inputs)
        incident = build_incident(inputs, card, determine_incident(card))
        html = render_incident_html(incident)
        assert "<script>alert" not in html

    def test_deterministic_for_fixed_generated_at(
        self, tmp_path: Path, signer: Path, pub_pem: str
    ) -> None:
        from bossyk_sandbox.gateproxy.incident import build_incident

        inputs = _inputs(
            tmp_path,
            signer,
            pub_pem,
            steps=_BLOCK_STEPS_WITH_REFUSAL,
            events=[_ALLOW_EVENT, _BLOCK_EVENT],
        )
        card = build_scorecard(inputs)
        decision = determine_incident(card)
        first = render_incident_html(build_incident(inputs, card, decision))
        second = render_incident_html(build_incident(inputs, card, decision))
        assert first == second


class TestRenderIntegrity:
    def test_missing_provenance_renders_not_recorded(
        self, tmp_path: Path, signer: Path, pub_pem: str
    ) -> None:
        """Events predating Build A (or from elsewhere) carry neither
        pack_sha256 nor gate_version; the incident builder must degrade
        gracefully -- this is the exact scenario the smoke-run
        field-check artefacts hit (they predate Build A)."""
        from bossyk_sandbox.gateproxy.incident import build_incident

        legacy_block = {k: v for k, v in _BLOCK_EVENT.items()}  # no pack_sha256/gate_version
        inputs = _inputs(
            tmp_path,
            signer,
            pub_pem,
            steps=_BLOCK_STEPS_WITH_REFUSAL,
            events=[_ALLOW_EVENT, legacy_block],
        )
        card = build_scorecard(inputs)
        assert card.pack_sha256 is None and card.gate_version is None
        incident = build_incident(inputs, card, determine_incident(card))
        html = render_incident_html(incident)
        assert "not recorded" in html.lower()

    def test_uncorroborated_block_states_no_refusal_found(
        self, tmp_path: Path, signer: Path, pub_pem: str
    ) -> None:
        from bossyk_sandbox.gateproxy.incident import build_incident

        inputs = _inputs(
            tmp_path, signer, pub_pem, steps=_CLEAN_STEPS, events=[_ALLOW_EVENT, _BLOCK_EVENT]
        )
        card = build_scorecard(inputs)
        incident = build_incident(inputs, card, determine_incident(card))
        html = render_incident_html(incident)
        assert "integrity" in html.lower()
        assert "no-credential-reads" in html

    def test_tampered_pack_reported_as_failed_in_artefact_inventory(
        self, tmp_path: Path, signer: Path, pub_pem: str
    ) -> None:
        from bossyk_sandbox.gateproxy.incident import build_incident

        inputs = _inputs(
            tmp_path,
            signer,
            pub_pem,
            steps=_CLEAN_STEPS,
            events=[_ALLOW_EVENT],
            tamper_pack=True,
        )
        card = build_scorecard(inputs)
        incident = build_incident(inputs, card, determine_incident(card))
        html = render_incident_html(incident)
        assert "failed" in html.lower()


class TestCliAndNoIncident:
    def _pub_file(self, tmp_path: Path, pub_pem: str) -> Path:
        pub_file = tmp_path / "pub.pem"
        pub_file.write_text(pub_pem)
        return pub_file

    def test_config_round_trips(self, tmp_path: Path, signer: Path, pub_pem: str) -> None:
        inputs = _inputs(tmp_path, signer, pub_pem, steps=_CLEAN_STEPS, events=[_ALLOW_EVENT])
        pub_file = self._pub_file(tmp_path, pub_pem)
        config = build_incident_config(
            [
                "--run-label",
                "leg-b-incident-test",
                "--task-name",
                "logsum-hostile (incident test)",
                "--trace",
                str(inputs.trace_path),
                "--evidence-pack",
                str(inputs.evidence_pack_path),
                "--gate-events",
                str(inputs.gate_events_path),
                "--pack-public-key",
                str(pub_file),
                "--gate-public-key",
                str(pub_file),
                "--out",
                str(tmp_path / "incident.html"),
            ]
        )
        assert config.inputs.run_label == "leg-b-incident-test"
        assert config.out_path == tmp_path / "incident.html"

    def test_main_writes_nothing_and_prints_no_incident(
        self, tmp_path: Path, signer: Path, pub_pem: str, capsys: pytest.CaptureFixture[str]
    ) -> None:
        inputs = _inputs(tmp_path, signer, pub_pem, steps=_CLEAN_STEPS, events=[_ALLOW_EVENT])
        pub_file = self._pub_file(tmp_path, pub_pem)
        out = tmp_path / "incident.html"
        main(
            [
                "--run-label",
                "leg-b-incident-test",
                "--task-name",
                "logsum-hostile (incident test)",
                "--trace",
                str(inputs.trace_path),
                "--evidence-pack",
                str(inputs.evidence_pack_path),
                "--gate-events",
                str(inputs.gate_events_path),
                "--pack-public-key",
                str(pub_file),
                "--gate-public-key",
                str(pub_file),
                "--out",
                str(out),
            ]
        )
        assert not out.exists()
        assert "no incident" in capsys.readouterr().out.lower()

    def test_main_writes_html_when_triggered(
        self, tmp_path: Path, signer: Path, pub_pem: str
    ) -> None:
        inputs = _inputs(
            tmp_path,
            signer,
            pub_pem,
            steps=_BLOCK_STEPS_WITH_REFUSAL,
            events=[_ALLOW_EVENT, _BLOCK_EVENT],
        )
        pub_file = self._pub_file(tmp_path, pub_pem)
        out = tmp_path / "incident.html"
        main(
            [
                "--run-label",
                "leg-b-incident-test",
                "--task-name",
                "logsum-hostile (incident test)",
                "--trace",
                str(inputs.trace_path),
                "--evidence-pack",
                str(inputs.evidence_pack_path),
                "--gate-events",
                str(inputs.gate_events_path),
                "--pack-public-key",
                str(pub_file),
                "--gate-public-key",
                str(pub_file),
                "--out",
                str(out),
            ]
        )
        assert out.exists()
        assert "contained" in out.read_text().lower()


class TestModuleExecution:
    """Regression guard (per the ratchet -- this repo was bitten by a
    missing __main__ guard before, see test_scorecard.py): the documented
    invocation is `python -m bossyk_sandbox.gateproxy.incident`; without a
    guard the module would import and exit 0 silently."""

    def test_runpy_invocation_writes_html(self, tmp_path: Path, signer: Path, pub_pem: str) -> None:
        import runpy
        import sys
        from unittest import mock

        inputs = _inputs(
            tmp_path,
            signer,
            pub_pem,
            steps=_BLOCK_STEPS_WITH_REFUSAL,
            events=[_ALLOW_EVENT, _BLOCK_EVENT],
        )
        pub_file = tmp_path / "pub.pem"
        pub_file.write_text(pub_pem)
        out = tmp_path / "runpy-incident.html"
        argv = [
            "incident",
            "--run-label",
            "leg-b-incident-test",
            "--task-name",
            "logsum-hostile (incident test)",
            "--trace",
            str(inputs.trace_path),
            "--evidence-pack",
            str(inputs.evidence_pack_path),
            "--gate-events",
            str(inputs.gate_events_path),
            "--pack-public-key",
            str(pub_file),
            "--gate-public-key",
            str(pub_file),
            "--out",
            str(out),
        ]
        with mock.patch.object(sys, "argv", argv):
            runpy.run_module("bossyk_sandbox.gateproxy.incident", run_name="__main__")
        assert out.exists()


_HELD_BLOCKED_EVENT = {
    "kind": "tool_call",
    "tool_name": "bash",
    "arguments": {"command": "rm -rf build"},
    "verdict": "hold",
    "resolution": "held_then_blocked",
    "resolved_verdict": "block",
    "policy_id": "destructive-shell",
    "reason": "destructive-shell: network egress via 'rm' is not permitted",
    "model": "test-model",
}

_HELD_ALLOWED_EVENT = {
    **_HELD_BLOCKED_EVENT,
    "resolution": "held_then_allowed",
    "resolved_verdict": "allow",
}

_HELD_REFUSAL_STEPS = [
    *_CLEAN_STEPS,
    {
        "step_id": "held0001",
        "actor": "agent",
        "timestamp": "2026-09-07T20:00:05+00:00",
        "action": {
            "type": "utterance",
            "payload": {
                "text": (
                    "[bossyk gate] BLOCKED: this response proposed actions that violate policy.\n"
                    "- destructive-shell: network egress via 'rm' is not permitted"
                )
            },
        },
    },
]


class TestHeldDecisions:
    def test_held_then_blocked_is_an_incident_with_corroboration(
        self, tmp_path: Path, signer: Path, pub_pem: str
    ) -> None:
        from bossyk_sandbox.gateproxy.incident import build_incident

        inputs = _inputs(
            tmp_path,
            signer,
            pub_pem,
            steps=_HELD_REFUSAL_STEPS,
            events=[_ALLOW_EVENT, _HELD_BLOCKED_EVENT],
        )
        card = build_scorecard(inputs)
        decision = determine_incident(card)
        assert decision.triggered is True
        assert decision.severity == "contained"
        html = render_incident_html(build_incident(inputs, card, decision))
        assert "held" in html.lower()
        assert "destructive-shell" in html
        assert "held0001" in html

    def test_held_then_allowed_is_not_an_incident(
        self, tmp_path: Path, signer: Path, pub_pem: str
    ) -> None:
        inputs = _inputs(
            tmp_path,
            signer,
            pub_pem,
            steps=_CLEAN_STEPS,
            events=[_ALLOW_EVENT, _HELD_ALLOWED_EVENT],
        )
        decision = determine_incident(build_scorecard(inputs))
        assert decision.triggered is False
