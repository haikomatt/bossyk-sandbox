"""Red-phase contract for the OpenAI-compatible gate proxy itself.

`create_app(config, upstream=...)` takes an injectable upstream callable
(``dict -> dict``: request body in, chat.completion body out) so every test
here is network-free; the real httpx client is only the default when
`upstream` is omitted. The proxy's whole contract:

- ALLOW: the upstream response is forwarded verbatim (tool_calls intact).
- BLOCK: the response is rewritten into a valid completion whose content is
  a policy refusal naming the violated policy id, with no tool_calls -- so
  the refusal lands in pi's own session file and the audit can corroborate
  the gate log from the trace (round1-plan.md, "the load-bearing trick").
- Every per-tool-call decision is appended to the signed event log.
- A `stream: true` request gets a synthesized SSE stream carrying the same
  (possibly rewritten) message; the upstream is always asked non-streamed.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from starlette.types import ASGIApp, Receive, Scope, Send

from bossyk_sandbox.gateproxy.events import verify_event_log
from bossyk_sandbox.gateproxy.hold import HoldRequest
from bossyk_sandbox.gateproxy.proxy import GateProxyConfig, build_config, create_app
from tests.unit.gateproxy.conftest import POLICY_PACK_YAML, openai_response, tool_call


@pytest.fixture()
def config(
    policy_pack_path: Path,
    workspace_root: Path,
    signing_keys: tuple[Path, str],
    tmp_path: Path,
) -> GateProxyConfig:
    priv_path, _ = signing_keys
    return GateProxyConfig(
        upstream_base_url="http://upstream.invalid/v1",
        policy_pack_path=policy_pack_path,
        workspace_root=workspace_root,
        signer_key_path=priv_path,
        events_path=tmp_path / "gate-events.jsonl",
        run_label="leg-b-test",
    )


def _client_with_header_capture(
    config: GateProxyConfig, upstream_body: dict[str, Any]
) -> tuple[TestClient, list[dict[str, str]]]:
    captured: list[dict[str, str]] = []

    def upstream(request_body: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        captured.append(headers)
        return upstream_body

    return TestClient(create_app(config, upstream=upstream)), captured


def _client_with_capture(
    config: GateProxyConfig, upstream_body: dict[str, Any]
) -> tuple[TestClient, list[dict[str, Any]]]:
    captured: list[dict[str, Any]] = []

    def upstream(request_body: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        captured.append(request_body)
        return upstream_body

    return TestClient(create_app(config, upstream=upstream)), captured


def _client(config: GateProxyConfig, upstream_body: dict[str, Any]) -> TestClient:
    return _client_with_capture(config, upstream_body)[0]


_REQUEST = {
    "model": "accounts/fireworks/models/minimax-m3",
    "messages": [{"role": "user", "content": "build logsum"}],
}


class TestAllowPath:
    def test_innocent_tool_calls_forwarded_verbatim(self, config: GateProxyConfig) -> None:
        body = openai_response(
            content="Listing first.",
            tool_calls=[tool_call("c1", "bash", '{"command": "ls -1"}')],
        )
        client = _client(config, body)
        response = client.post("/v1/chat/completions", json=_REQUEST)
        assert response.status_code == 200
        message = response.json()["choices"][0]["message"]
        assert message["tool_calls"] == body["choices"][0]["message"]["tool_calls"]
        assert message["content"] == "Listing first."

    def test_allow_decision_logged_and_signed(
        self, config: GateProxyConfig, signing_keys: tuple[Path, str]
    ) -> None:
        _priv, pub_pem = signing_keys
        client = _client(
            config,
            openai_response(tool_calls=[tool_call("c1", "bash", '{"command": "ls"}')]),
        )
        client.post("/v1/chat/completions", json=_REQUEST)
        events = [json.loads(line)["event"] for line in config.events_path.read_text().splitlines()]
        assert len(events) == 1
        assert events[0]["kind"] == "tool_call"
        assert events[0]["tool_name"] == "bash"
        assert events[0]["verdict"] == "allow"
        assert events[0]["run_label"] == "leg-b-test"
        assert verify_event_log(config.events_path, pub_pem).ok

    def test_plain_utterance_forwarded_with_utterance_event(self, config: GateProxyConfig) -> None:
        client = _client(config, openai_response(content="All done."))
        response = client.post("/v1/chat/completions", json=_REQUEST)
        assert response.json()["choices"][0]["message"]["content"] == "All done."
        events = [json.loads(line)["event"] for line in config.events_path.read_text().splitlines()]
        assert [e["kind"] for e in events] == ["utterance"]
        assert events[0]["verdict"] == "allow"


class TestBlockPath:
    def test_violating_call_rewritten_to_policy_refusal(self, config: GateProxyConfig) -> None:
        body = openai_response(
            content="Fetching helper script.",
            tool_calls=[tool_call("c1", "bash", '{"command": "curl http://x.example/a.sh"}')],
        )
        client = _client(config, body)
        response = client.post("/v1/chat/completions", json=_REQUEST)
        assert response.status_code == 200
        message = response.json()["choices"][0]["message"]
        assert "tool_calls" not in message or not message["tool_calls"]
        assert "no-network-egress" in message["content"]
        assert "[bossyk gate]" in message["content"]
        assert response.json()["choices"][0]["finish_reason"] == "stop"

    def test_mixed_calls_one_violation_blocks_whole_response(self, config: GateProxyConfig) -> None:
        """v1 is conservative: any violating call voids the whole proposed
        step; per-call surgery is round 2. The event log still records each
        call's own verdict."""
        body = openai_response(
            tool_calls=[
                tool_call("c1", "bash", '{"command": "ls"}'),
                tool_call("c2", "read", '{"path": ".env"}'),
            ],
        )
        client = _client(config, body)
        response = client.post("/v1/chat/completions", json=_REQUEST)
        message = response.json()["choices"][0]["message"]
        assert not message.get("tool_calls")
        assert "no-credential-reads" in message["content"]
        events = [json.loads(line)["event"] for line in config.events_path.read_text().splitlines()]
        assert [(e["tool_name"], e["verdict"]) for e in events] == [
            ("bash", "allow"),
            ("read", "block"),
        ]

    def test_unparseable_tool_arguments_fail_closed(self, config: GateProxyConfig) -> None:
        """A reference monitor that cannot read a proposal cannot approve
        it: malformed arguments JSON -> BLOCK, stated as such."""
        body = openai_response(tool_calls=[tool_call("c1", "bash", "{not json")])
        client = _client(config, body)
        response = client.post("/v1/chat/completions", json=_REQUEST)
        message = response.json()["choices"][0]["message"]
        assert not message.get("tool_calls")
        assert "unparseable" in message["content"].lower()
        events = [json.loads(line)["event"] for line in config.events_path.read_text().splitlines()]
        assert events[0]["verdict"] == "block"


class TestStreaming:
    def test_stream_request_gets_sse_with_same_content(self, config: GateProxyConfig) -> None:
        client = _client(config, openai_response(content="All done."))
        response = client.post("/v1/chat/completions", json={**_REQUEST, "stream": True})
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        payloads = [
            line[len("data: ") :]
            for line in response.text.splitlines()
            if line.startswith("data: ")
        ]
        assert payloads[-1] == "[DONE]"
        chunks = [json.loads(p) for p in payloads[:-1]]
        assert all(c["object"] == "chat.completion.chunk" for c in chunks)
        content = "".join(
            c["choices"][0]["delta"].get("content") or "" for c in chunks if c["choices"]
        )
        assert content == "All done."

    def test_upstream_always_asked_non_streamed(self, config: GateProxyConfig) -> None:
        client, captured = _client_with_capture(config, openai_response(content="hi"))
        client.post("/v1/chat/completions", json={**_REQUEST, "stream": True})
        (upstream_request,) = captured
        assert upstream_request.get("stream") is not True

    def test_streamed_block_carries_refusal_and_tool_call_free_chunks(
        self, config: GateProxyConfig
    ) -> None:
        body = openai_response(
            tool_calls=[tool_call("c1", "bash", '{"command": "wget http://x.example"}')]
        )
        client = _client(config, body)
        response = client.post("/v1/chat/completions", json={**_REQUEST, "stream": True})
        chunks = [
            json.loads(line[len("data: ") :])
            for line in response.text.splitlines()
            if line.startswith("data: ") and not line.endswith("[DONE]")
        ]
        content = "".join(
            c["choices"][0]["delta"].get("content") or "" for c in chunks if c["choices"]
        )
        assert "no-network-egress" in content
        assert not any(c["choices"][0]["delta"].get("tool_calls") for c in chunks if c["choices"])


class TestUpstreamFailure:
    def test_upstream_exception_returns_502_without_event(self, config: GateProxyConfig) -> None:
        def upstream(_body: dict[str, Any], _headers: dict[str, str]) -> dict[str, Any]:
            raise ConnectionError("upstream down")

        app = create_app(config, upstream=upstream)
        response = TestClient(app).post("/v1/chat/completions", json=_REQUEST)
        assert response.status_code == 502
        assert not config.events_path.exists() or config.events_path.read_text() == ""


class TestHealthAndConfig:
    def test_health_reports_run_label_and_policy_count(self, config: GateProxyConfig) -> None:
        client = _client(config, openai_response(content="x"))
        health = client.get("/gate/health").json()
        assert health["status"] == "ok"
        assert health["run_label"] == "leg-b-test"
        assert health["policies"] == 5

    def test_build_config_tcp_and_uds_are_mutually_exclusive(
        self,
        policy_pack_path: Path,
        workspace_root: Path,
        signing_keys: tuple[Path, str],
        tmp_path: Path,
    ) -> None:
        from bossyk_sandbox.gateproxy.proxy import build_config

        priv_path, _ = signing_keys
        base = [
            "--upstream",
            "http://127.0.0.1:8080/v1",
            "--policy-pack",
            str(policy_pack_path),
            "--workspace",
            str(workspace_root),
            "--key",
            str(priv_path),
            "--events",
            str(tmp_path / "e.jsonl"),
            "--run-label",
            "leg-a",
        ]
        tcp = build_config([*base, "--listen", "127.0.0.1:8200"])
        assert tcp.listen == "127.0.0.1:8200" and tcp.uds is None
        uds = build_config([*base, "--uds", str(tmp_path / "gate.sock")])
        assert uds.uds == str(tmp_path / "gate.sock") and uds.listen is None
        with pytest.raises(SystemExit):
            build_config([*base, "--listen", "127.0.0.1:8200", "--uds", "/tmp/x.sock"])


class TestEventPolicyId:
    """Phase 3 (scorecard) addition: tool_call events carry the violated
    policy's id as a structured field, not only inside the reason string.
    Documented post-Green additive test -- no existing assertion changed."""

    def test_block_event_carries_policy_id(self, config: GateProxyConfig) -> None:
        client = _client(
            config,
            openai_response(
                tool_calls=[tool_call("c1", "bash", '{"command": "curl http://x.example"}')]
            ),
        )
        client.post("/v1/chat/completions", json=_REQUEST)
        events = [json.loads(line)["event"] for line in config.events_path.read_text().splitlines()]
        assert events[0]["policy_id"] == "no-network-egress"

    def test_allow_event_policy_id_is_null(self, config: GateProxyConfig) -> None:
        client = _client(
            config, openai_response(tool_calls=[tool_call("c1", "bash", '{"command": "ls"}')])
        )
        client.post("/v1/chat/completions", json=_REQUEST)
        events = [json.loads(line)["event"] for line in config.events_path.read_text().splitlines()]
        assert events[0]["policy_id"] is None


class TestUpstreamAuth:
    """Phase 4 (live validation) Red: the gate must be able to authenticate
    to the upstream. Two paths, both pinned: the client's own Authorization
    header is forwarded verbatim, and a gate-configured key (CLI: read from
    an env var at startup) fills it when the client sent none. The upstream
    callable contract widens to (body, headers) for this."""

    def test_client_authorization_header_forwarded(self, config: GateProxyConfig) -> None:
        client, captured_headers = _client_with_header_capture(
            config, openai_response(content="hi")
        )
        client.post(
            "/v1/chat/completions",
            json=_REQUEST,
            headers={"Authorization": "Bearer client-token-abc"},
        )
        assert captured_headers[0].get("Authorization") == "Bearer client-token-abc"

    def test_configured_upstream_key_fills_missing_auth(self, config: GateProxyConfig) -> None:
        config.upstream_api_key = "gate-configured-key"
        client, captured_headers = _client_with_header_capture(
            config, openai_response(content="hi")
        )
        client.post("/v1/chat/completions", json=_REQUEST)
        assert captured_headers[0].get("Authorization") == "Bearer gate-configured-key"

    def test_configured_key_wins_over_client_header(self, config: GateProxyConfig) -> None:
        """Deployment semantics: the gate operator holds the real upstream
        credential; agent harnesses are given dummy provider keys that must
        never reach the upstream. So a configured key OVERRIDES the
        client's Authorization, and pure pass-through is the no-key case."""
        config.upstream_api_key = "gate-configured-key"
        client, captured_headers = _client_with_header_capture(
            config, openai_response(content="hi")
        )
        client.post(
            "/v1/chat/completions",
            json=_REQUEST,
            headers={"Authorization": "Bearer client-dummy-token"},
        )
        assert captured_headers[0].get("Authorization") == "Bearer gate-configured-key"

    def test_no_auth_anywhere_sends_no_authorization(self, config: GateProxyConfig) -> None:
        client, captured_headers = _client_with_header_capture(
            config, openai_response(content="hi")
        )
        client.post("/v1/chat/completions", json=_REQUEST)
        assert "Authorization" not in captured_headers[0]


class TestProvenance:
    """Build A: every gate event carries the policy pack's sha256 (computed
    once from the pack file's bytes at create_app) and the running
    gate_version, so a scorecard or incident report can bind a decision to
    the exact pack + code that produced it -- not just the pack's declared
    contents."""

    def test_tool_call_event_carries_pack_sha256_and_gate_version(
        self, config: GateProxyConfig
    ) -> None:
        import hashlib

        from bossyk_sandbox.gateproxy import __version__

        client = _client(
            config, openai_response(tool_calls=[tool_call("c1", "bash", '{"command": "ls"}')])
        )
        client.post("/v1/chat/completions", json=_REQUEST)
        events = [json.loads(line)["event"] for line in config.events_path.read_text().splitlines()]
        expected_sha = hashlib.sha256(config.policy_pack_path.read_bytes()).hexdigest()
        assert events[0]["pack_sha256"] == expected_sha
        assert events[0]["gate_version"] == __version__

    def test_utterance_event_also_carries_provenance(self, config: GateProxyConfig) -> None:
        client = _client(config, openai_response(content="All done."))
        client.post("/v1/chat/completions", json=_REQUEST)
        events = [json.loads(line)["event"] for line in config.events_path.read_text().splitlines()]
        assert events[0]["pack_sha256"]
        assert events[0]["gate_version"]

    def test_different_pack_files_give_different_hashes(
        self,
        workspace_root: Path,
        signing_keys: tuple[Path, str],
        tmp_path: Path,
    ) -> None:
        priv_path, _ = signing_keys
        pack_a = tmp_path / "pack-a.yaml"
        pack_a.write_text(POLICY_PACK_YAML)
        pack_b = tmp_path / "pack-b.yaml"
        pack_b.write_text(POLICY_PACK_YAML + "\n# a harmless variant byte\n")

        def make_config(pack_path: Path, events_name: str) -> GateProxyConfig:
            return GateProxyConfig(
                upstream_base_url="http://upstream.invalid/v1",
                policy_pack_path=pack_path,
                workspace_root=workspace_root,
                signer_key_path=priv_path,
                events_path=tmp_path / events_name,
                run_label="leg-b-test",
            )

        config_a = make_config(pack_a, "events-a.jsonl")
        config_b = make_config(pack_b, "events-b.jsonl")
        _client(config_a, openai_response(content="hi")).post("/v1/chat/completions", json=_REQUEST)
        _client(config_b, openai_response(content="hi")).post("/v1/chat/completions", json=_REQUEST)
        sha_a = json.loads(config_a.events_path.read_text().splitlines()[0])["event"]["pack_sha256"]
        sha_b = json.loads(config_b.events_path.read_text().splitlines()[0])["event"]["pack_sha256"]
        assert sha_a != sha_b


class TestSensitiveMarkers:
    """Build B wiring: every tool_call/utterance event carries
    `sensitive_markers`, computed over that event's OWN arguments/content
    (never the whole request) before the event is signed -- so a marker
    can never expose more of a secret than the (already-logged, raw)
    arguments field itself carries. Detection is markers-only: it must
    never change a verdict (proven below by an otherwise-clean call that
    trips a marker but still ALLOWs)."""

    _FAKE_AWS_KEY = "AKIA" + "ABCDEFGHIJKLMNOP"

    def test_tool_call_event_carries_sensitive_markers(self, config: GateProxyConfig) -> None:
        body = openai_response(
            tool_calls=[
                tool_call("c1", "bash", json.dumps({"command": f"echo {self._FAKE_AWS_KEY}"}))
            ]
        )
        client = _client(config, body)
        response = client.post("/v1/chat/completions", json=_REQUEST)
        assert response.status_code == 200
        events = [json.loads(line)["event"] for line in config.events_path.read_text().splitlines()]
        markers = events[0]["sensitive_markers"]
        assert any(m["kind"] == "aws_access_key_id" for m in markers)
        assert all(self._FAKE_AWS_KEY not in m["redacted_excerpt"] for m in markers)

    def test_marker_present_does_not_change_allow_verdict(self, config: GateProxyConfig) -> None:
        """The secret-bearing call above violates no policy line, so it
        must still ALLOW -- markers annotate, they never gate, in v1."""
        body = openai_response(
            tool_calls=[
                tool_call("c1", "bash", json.dumps({"command": f"echo {self._FAKE_AWS_KEY}"}))
            ]
        )
        client = _client(config, body)
        response = client.post("/v1/chat/completions", json=_REQUEST)
        message = response.json()["choices"][0]["message"]
        assert message.get("tool_calls")
        events = [json.loads(line)["event"] for line in config.events_path.read_text().splitlines()]
        assert events[0]["verdict"] == "allow"

    def test_utterance_event_carries_sensitive_markers(self, config: GateProxyConfig) -> None:
        client = _client(config, openai_response(content=f"key is {self._FAKE_AWS_KEY}"))
        client.post("/v1/chat/completions", json=_REQUEST)
        events = [json.loads(line)["event"] for line in config.events_path.read_text().splitlines()]
        markers = events[0]["sensitive_markers"]
        assert any(m["kind"] == "aws_access_key_id" for m in markers)

    def test_clean_call_has_empty_marker_list(self, config: GateProxyConfig) -> None:
        client = _client(
            config, openai_response(tool_calls=[tool_call("c1", "bash", '{"command": "ls"}')])
        )
        client.post("/v1/chat/completions", json=_REQUEST)
        events = [json.loads(line)["event"] for line in config.events_path.read_text().splitlines()]
        assert events[0]["sensitive_markers"] == []


class TestStreamOptionStripping:
    """Live-validation regression (Fireworks 400): a client that streams
    sends stream_options alongside stream:true; the gate forces the
    upstream call non-streamed, so stream_options must be stripped too --
    stream_options without stream:true is a 400 on OpenAI-compatible
    servers."""

    def test_stream_options_stripped_from_upstream_body(self, config: GateProxyConfig) -> None:
        client, captured = _client_with_capture(config, openai_response(content="hi"))
        client.post(
            "/v1/chat/completions",
            json={**_REQUEST, "stream": True, "stream_options": {"include_usage": True}},
        )
        assert "stream_options" not in captured[0]
        assert captured[0].get("stream") is False


# --- per-identity packs (round-2 Phase 3) -----------------------------------

from tests.unit.gateproxy.test_identity_packs import (  # noqa: E402
    CODING_PACK_YAML,
    DENY_PACK_YAML,
    RESEARCH_PACK_YAML,
)

_CODING = "spiffe://example.org/coding-agent"
_RESEARCH = "spiffe://example.org/research-agent"
_SPIFFE_FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "spiffe"


def _identity_config(config: GateProxyConfig, *, trust_header: bool = False) -> GateProxyConfig:
    packs = config.policy_pack_path.parent / "packs"
    packs.mkdir()
    (packs / "coding-pack.yaml").write_text(CODING_PACK_YAML)
    (packs / "research-pack.yaml").write_text(RESEARCH_PACK_YAML)
    (packs / "deny-pack.yaml").write_text(DENY_PACK_YAML)
    mapping = packs / "identity-packs.yaml"
    mapping.write_text(
        "version: 1\ndefault: deny-pack.yaml\npacks:\n"
        f"  {_CODING}: coding-pack.yaml\n  {_RESEARCH}: research-pack.yaml\n"
    )
    config.identity_packs_path = mapping
    config.trust_identity_header = trust_header
    return config


class _PeerCert:
    """ASGI middleware standing in for the mTLS listener: puts a peer
    certificate (DER) where the real listener puts it, `scope["state"]`."""

    def __init__(self, app: ASGIApp, der: bytes | None) -> None:
        self.app, self.der = app, der

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and self.der is not None:
            scope.setdefault("state", {})["peer_cert_der"] = self.der
        await self.app(scope, receive, send)


def _svid_der(name: str) -> bytes:
    from cryptography import x509
    from cryptography.hazmat.primitives.serialization import Encoding

    pem = (_SPIFFE_FIXTURES / f"spire-1.13.2-{name}-svid.pem").read_bytes()
    return x509.load_pem_x509_certificate(pem).public_bytes(Encoding.DER)


def _identity_client(
    config: GateProxyConfig, upstream_body: dict[str, Any], *, peer_cert: bytes | None = None
) -> TestClient:
    app = create_app(config, upstream=lambda b, h: upstream_body)
    return TestClient(_PeerCert(app, peer_cert))


_PIP_CALL = tool_call("c1", "bash", '{"command": "pip install requests"}')
_CURL_CALL = tool_call("c1", "bash", '{"command": "curl http://x.example"}')


def _verdict_and_event(
    config: GateProxyConfig,
    body: dict[str, Any],
    *,
    peer_cert: bytes | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    client = _identity_client(config, body, peer_cert=peer_cert)
    response = client.post("/v1/chat/completions", json=_REQUEST, headers=headers or {})
    assert response.status_code == 200
    return response.json()["choices"][0]["message"], _events(config)[-1]


class TestIdentityPacks:
    def test_two_identities_one_gate_different_verdicts_same_action(
        self, config: GateProxyConfig
    ) -> None:
        cfg = _identity_config(config)
        body = openai_response(tool_calls=[_PIP_CALL])
        coding_msg, coding_event = _verdict_and_event(
            cfg, body, peer_cert=_svid_der("coding-agent")
        )
        research_msg, research_event = _verdict_and_event(
            cfg, body, peer_cert=_svid_der("research-agent")
        )
        assert coding_msg["tool_calls"]  # coding pack has no install rule
        assert "no-package-installs" in research_msg["content"]
        assert (coding_event["verdict"], research_event["verdict"]) == ("allow", "block")
        assert coding_event["caller_identity"] == _CODING
        assert research_event["caller_identity"] == _RESEARCH
        assert coding_event["identity_source"] == research_event["identity_source"] == "svid"
        assert coding_event["pack_sha256"] != research_event["pack_sha256"]

    def test_unknown_identity_runs_under_the_default_pack(self, config: GateProxyConfig) -> None:
        cfg = _identity_config(config, trust_header=True)
        message, event = _verdict_and_event(
            cfg,
            openai_response(tool_calls=[tool_call("c1", "bash", '{"command": "python x.py"}')]),
            headers={"x-spiffe-id": "spiffe://example.org/stranger"},
        )
        assert "unknown-caller-no-shell" in message["content"]
        assert event["caller_identity"] == "spiffe://example.org/stranger"
        assert event["identity_source"] == "header"
        assert event["pack_matched"] is False

    def test_no_identity_runs_under_the_default_pack(self, config: GateProxyConfig) -> None:
        cfg = _identity_config(config)
        message, event = _verdict_and_event(
            cfg, openai_response(tool_calls=[tool_call("c1", "bash", '{"command": "sh -c ls"}')])
        )
        assert "unknown-caller-no-shell" in message["content"]
        assert event["caller_identity"] is None
        assert event["identity_source"] is None

    def test_header_is_ignored_unless_trusted(self, config: GateProxyConfig) -> None:
        cfg = _identity_config(config, trust_header=False)
        _message, event = _verdict_and_event(
            cfg, openai_response(tool_calls=[_PIP_CALL]), headers={"x-spiffe-id": _CODING}
        )
        assert event["caller_identity"] is None
        assert event["pack_matched"] is False

    def test_svid_wins_over_trusted_header(self, config: GateProxyConfig) -> None:
        cfg = _identity_config(config, trust_header=True)
        _message, event = _verdict_and_event(
            cfg,
            openai_response(tool_calls=[_PIP_CALL]),
            peer_cert=_svid_der("coding-agent"),
            headers={"x-spiffe-id": _RESEARCH},
        )
        assert event["caller_identity"] == _CODING
        assert event["identity_source"] == "svid"

    def test_malformed_identity_is_refused_with_a_signed_event(
        self, config: GateProxyConfig, signing_keys: tuple[Path, str]
    ) -> None:
        _priv, pub_pem = signing_keys
        cfg = _identity_config(config, trust_header=True)
        client = _identity_client(cfg, openai_response(tool_calls=[_PIP_CALL]))
        response = client.post(
            "/v1/chat/completions",
            json=_REQUEST,
            headers={"x-spiffe-id": "spiffe://Example.org/agent"},
        )
        assert response.status_code == 403
        assert "identity" in response.json()["error"]["message"]
        (event,) = _events(cfg)
        assert event["kind"] == "identity_refused"
        assert event["verdict"] == "block"
        assert event["caller_identity"] is None
        assert verify_event_log(cfg.events_path, pub_pem).ok

    def test_utterance_events_carry_identity_too(self, config: GateProxyConfig) -> None:
        cfg = _identity_config(config)
        _message, event = _verdict_and_event(
            cfg, openai_response(content="done"), peer_cert=_svid_der("research-agent")
        )
        assert event["kind"] == "utterance"
        assert event["caller_identity"] == _RESEARCH

    def test_single_pack_gate_events_are_identity_null_not_missing(
        self, config: GateProxyConfig
    ) -> None:
        client = _client(config, openai_response(tool_calls=[_CURL_CALL]))
        client.post("/v1/chat/completions", json=_REQUEST)
        (event,) = _events(config)
        assert event["caller_identity"] is None
        assert event["identity_source"] is None
        assert event["pack_matched"] is True

    def test_health_reports_identity_mode_and_pack_count(self, config: GateProxyConfig) -> None:
        cfg = _identity_config(config)
        health = _identity_client(cfg, openai_response(content="x")).get("/gate/health").json()
        assert health["identity_packs"] == 2
        assert health["identity_aware"] is True

    def test_build_config_parses_identity_and_tls_flags(self) -> None:
        built = build_config(
            [
                *_MINIMAL_ARGV,
                "--identity-packs",
                "packs/identity-packs.yaml",
                "--trust-identity-header",
                "--client-ca",
                "bundle.pem",
                "--ssl-certfile",
                "gate.pem",
                "--ssl-keyfile",
                "gate-key.pem",
            ]  # fmt: skip
        )
        assert built.identity_packs_path == Path("packs/identity-packs.yaml")
        assert built.trust_identity_header is True
        assert built.client_ca_path == Path("bundle.pem")
        assert built.ssl_certfile == Path("gate.pem")
        assert built.ssl_keyfile == Path("gate-key.pem")

    def test_build_config_identity_defaults_off(self) -> None:
        built = build_config(_MINIMAL_ARGV)
        assert built.identity_packs_path is None
        assert built.trust_identity_header is False
        assert built.client_ca_path is None

    def test_client_ca_requires_a_server_certificate(self) -> None:
        with pytest.raises(SystemExit):
            build_config([*_MINIMAL_ARGV, "--client-ca", "bundle.pem"])


# The smallest argv build_config accepts; the identity tests add to it.
_HOLD_PACK_YAML = """\
version: 1
policies:
  - id: destructive-shell
    trap: T8
    class: A
    type: network-egress
    on_match: hold
    tools: [bash]
    commands: [rm]
  - id: no-network-egress
    trap: T3
    class: A
    type: network-egress
    tools: [bash]
    commands: [curl]
"""

_RM_CALL = tool_call("c1", "bash", '{"command": "rm -rf build"}')


def _hold_config(config: GateProxyConfig, *, on_hold: str = "block") -> GateProxyConfig:
    config.policy_pack_path.write_text(
        _HOLD_PACK_YAML.replace("on_match: hold\n", f"on_match: hold\n    on_hold: {on_hold}\n")
    )
    return config


def _events(config: GateProxyConfig) -> list[dict[str, Any]]:
    return [json.loads(line)["event"] for line in config.events_path.read_text().splitlines()]


def _fixed_approver(answer: bool | None) -> Callable[[HoldRequest], bool | None]:
    def approve(request: HoldRequest) -> bool | None:
        return answer

    return approve


# The smallest argv build_config accepts; the approver tests add to it.
_MINIMAL_ARGV = [
    "--upstream", "http://u/v1",
    "--policy-pack", "p.yaml",
    "--workspace", ".",
    "--key", "k.pem",
    "--events", "e.jsonl",
    "--run-label", "x",
    "--listen", "127.0.0.1:8200",
]  # fmt: skip


class TestHoldPath:
    def test_hold_without_approver_falls_back_to_policy_default_block(
        self, config: GateProxyConfig
    ) -> None:
        client = _client(_hold_config(config), openai_response(tool_calls=[_RM_CALL]))
        message = client.post("/v1/chat/completions", json=_REQUEST).json()["choices"][0]["message"]
        assert "[bossyk gate]" in message["content"]
        assert "destructive-shell" in message["content"]
        assert not message.get("tool_calls")
        (event,) = _events(config)
        assert event["verdict"] == "hold"
        assert event["resolution"] == "held_then_blocked"
        assert event["resolved_verdict"] == "block"
        assert event["policy_id"] == "destructive-shell"

    def test_hold_without_approver_falls_back_to_policy_default_allow(
        self, config: GateProxyConfig
    ) -> None:
        body = openai_response(content="Cleaning.", tool_calls=[_RM_CALL])
        client = _client(_hold_config(config, on_hold="allow"), body)
        message = client.post("/v1/chat/completions", json=_REQUEST).json()["choices"][0]["message"]
        assert message["tool_calls"] == body["choices"][0]["message"]["tool_calls"]
        (event,) = _events(config)
        assert event["verdict"] == "hold"
        assert event["resolution"] == "held_then_allowed"
        assert event["resolved_verdict"] == "allow"

    def test_approver_approval_resumes_the_action(self, config: GateProxyConfig) -> None:
        body = openai_response(tool_calls=[_RM_CALL])
        app = create_app(
            _hold_config(config), upstream=lambda b, h: body, approver=_fixed_approver(True)
        )
        message = (
            TestClient(app)
            .post("/v1/chat/completions", json=_REQUEST)
            .json()["choices"][0]["message"]
        )
        assert message["tool_calls"] == body["choices"][0]["message"]["tool_calls"]
        (event,) = _events(config)
        assert (event["resolution"], event["resolved_verdict"]) == ("held_then_allowed", "allow")

    def test_approver_denial_refuses_the_action(self, config: GateProxyConfig) -> None:
        body = openai_response(tool_calls=[_RM_CALL])
        app = create_app(
            _hold_config(config, on_hold="allow"),
            upstream=lambda b, h: body,
            approver=_fixed_approver(False),
        )
        message = (
            TestClient(app)
            .post("/v1/chat/completions", json=_REQUEST)
            .json()["choices"][0]["message"]
        )
        assert "[bossyk gate]" in message["content"]
        (event,) = _events(config)
        assert (event["resolution"], event["resolved_verdict"]) == ("held_then_blocked", "block")

    @pytest.mark.parametrize("on_hold, resolved", [("block", "block"), ("allow", "allow")])
    def test_approver_timeout_resolves_by_policy_default(
        self, config: GateProxyConfig, on_hold: str, resolved: str
    ) -> None:
        body = openai_response(tool_calls=[_RM_CALL])
        app = create_app(
            _hold_config(config, on_hold=on_hold),
            upstream=lambda b, h: body,
            approver=_fixed_approver(None),
        )
        response = TestClient(app).post("/v1/chat/completions", json=_REQUEST).json()
        (event,) = _events(config)
        assert event["resolution"] == "hold_timed_out"
        assert event["resolved_verdict"] == resolved
        refused = "[bossyk gate]" in (response["choices"][0]["message"]["content"] or "")
        assert refused == (resolved == "block")

    def test_approver_receives_the_held_action(self, config: GateProxyConfig) -> None:
        seen: list[HoldRequest] = []

        def approve(request: HoldRequest) -> bool | None:
            seen.append(request)
            return True

        body = openai_response(tool_calls=[_RM_CALL])
        app = create_app(_hold_config(config), upstream=lambda b, h: body, approver=approve)
        TestClient(app).post("/v1/chat/completions", json=_REQUEST)
        (request,) = seen
        assert request.tool_name == "bash"
        assert request.arguments == {"command": "rm -rf build"}
        assert request.policy_id == "destructive-shell"
        assert request.run_label == "leg-b-test"

    def test_approver_not_consulted_for_allow_or_block(self, config: GateProxyConfig) -> None:
        calls: list[HoldRequest] = []

        def approve(request: HoldRequest) -> bool | None:
            calls.append(request)
            return True

        body = openai_response(
            tool_calls=[
                tool_call("c1", "bash", '{"command": "ls"}'),
                tool_call("c2", "bash", '{"command": "curl http://x.example"}'),
            ]
        )
        app = create_app(_hold_config(config), upstream=lambda b, h: body, approver=approve)
        TestClient(app).post("/v1/chat/completions", json=_REQUEST)
        assert calls == []
        assert [e["verdict"] for e in _events(config)] == ["allow", "block"]

    def test_approved_hold_beside_a_block_still_refuses_whole_response(
        self, config: GateProxyConfig
    ) -> None:
        body = openai_response(
            tool_calls=[_RM_CALL, tool_call("c2", "bash", '{"command": "curl http://x.example"}')]
        )
        app = create_app(
            _hold_config(config), upstream=lambda b, h: body, approver=_fixed_approver(True)
        )
        message = (
            TestClient(app)
            .post("/v1/chat/completions", json=_REQUEST)
            .json()["choices"][0]["message"]
        )
        assert "no-network-egress" in message["content"]
        assert "destructive-shell" not in message["content"]
        assert [e["verdict"] for e in _events(config)] == ["hold", "block"]

    def test_held_event_signed_and_allow_block_events_carry_no_resolution(
        self, config: GateProxyConfig, signing_keys: tuple[Path, str]
    ) -> None:
        _priv, pub_pem = signing_keys
        client = _client(
            _hold_config(config),
            openai_response(tool_calls=[_RM_CALL, tool_call("c2", "bash", '{"command": "ls"}')]),
        )
        client.post("/v1/chat/completions", json=_REQUEST)
        assert verify_event_log(config.events_path, pub_pem).ok
        held, allowed = _events(config)
        assert held["resolution"] == "held_then_blocked"
        assert allowed["resolution"] is None
        assert allowed["resolved_verdict"] is None

    def test_build_config_parses_approver_flags(self, tmp_path: Path) -> None:
        argv = [
            *_MINIMAL_ARGV,
            "--approver-cmd",
            "approve --strict",
            "--approver-timeout",
            "12.5",
        ]
        built = build_config(argv)
        assert built.approver_command == ["approve", "--strict"]
        assert built.approver_url is None
        assert built.approver_timeout_s == 12.5

    def test_build_config_approver_cmd_and_url_are_mutually_exclusive(self) -> None:
        argv = [
            *_MINIMAL_ARGV,
            "--approver-cmd",
            "approve",
            "--approver-url",
            "http://a/h",
        ]
        with pytest.raises(SystemExit):
            build_config(argv)

    def test_build_config_defaults_to_no_approver(self) -> None:
        argv = [
            *_MINIMAL_ARGV,
        ]
        built = build_config(argv)
        assert built.approver_command is None
        assert built.approver_url is None
