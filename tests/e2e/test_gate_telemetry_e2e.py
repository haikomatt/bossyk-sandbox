"""End-to-end: the gate's telemetry against a REAL OpenTelemetry collector
and a REAL Prometheus push gateway.

Gated on RUN_GATE_TELEMETRY_E2E=1. Point it at the receivers with
BOSSYK_E2E_OTLP_ENDPOINT (default http://127.0.0.1:14318) and
BOSSYK_E2E_PUSHGATEWAY_URL (default http://127.0.0.1:19091), e.g.

    docker run -d --name otel -p 127.0.0.1:14318:4318 \\
        -v $PWD/collector.yaml:/etc/otelcol/config.yaml:ro \\
        otel/opentelemetry-collector --config /etc/otelcol/config.yaml
    docker run -d --name pgw -p 127.0.0.1:19091:9091 prom/pushgateway

The push gateway is read back over HTTP, so the metric side is asserted
end to end; the collector side is asserted as "accepted" (HTTP 200 on
every OTLP request) since a debug exporter's stdout is not reachable
from here. The signed event log is verified afterwards to show telemetry
never touches the system of record.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from bossyk_sandbox.gateproxy.events import verify_event_log
from bossyk_sandbox.gateproxy.proxy import GateProxyConfig, create_app
from tests.unit.gateproxy.conftest import openai_response, tool_call

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_GATE_TELEMETRY_E2E") != "1",
    reason="Set RUN_GATE_TELEMETRY_E2E=1 with a collector and pushgateway running.",
)

_OTLP = os.environ.get("BOSSYK_E2E_OTLP_ENDPOINT", "http://127.0.0.1:14318")
_PGW = os.environ.get("BOSSYK_E2E_PUSHGATEWAY_URL", "http://127.0.0.1:19091")


def test_decisions_reach_real_collector_and_pushgateway(
    policy_pack_path: Path,
    workspace_root: Path,
    signing_keys: tuple[Path, str],
    tmp_path: Path,
) -> None:
    priv, pub_pem = signing_keys
    run_label = f"e2e-{tmp_path.name}"
    config = GateProxyConfig(
        upstream_base_url="http://upstream.invalid/v1",
        policy_pack_path=policy_pack_path,
        workspace_root=workspace_root,
        signer_key_path=priv,
        events_path=tmp_path / "gate-events.jsonl",
        run_label=run_label,
        otlp_endpoint=_OTLP,
        pushgateway_url=_PGW,
    )
    body = openai_response(
        tool_calls=[
            tool_call("c1", "bash", '{"command": "ls"}'),
            tool_call("c2", "bash", '{"command": "curl http://x.example"}'),
        ]
    )

    def upstream(request_body: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        return body

    client = TestClient(create_app(config, upstream=upstream))
    response = client.post("/v1/chat/completions", json={"model": "m", "messages": []})
    assert response.status_code == 200

    scraped = httpx.get(f"{_PGW}/metrics", timeout=10.0).text
    assert (
        f'bossyk_gate_decisions_total{{instance="",job="bossyk-gate",policy_id="",'
        f'run_label="{run_label}",verdict="allow"}} 1'
    ) in scraped
    assert (
        f'bossyk_gate_decisions_total{{instance="",job="bossyk-gate",'
        f'policy_id="no-network-egress",run_label="{run_label}",verdict="block"}} 1'
    ) in scraped
    assert client.get("/gate/metrics").text.count("bossyk_gate_decisions_total{") == 2

    events = [json.loads(line)["event"] for line in config.events_path.read_text().splitlines()]
    assert [e["verdict"] for e in events] == ["allow", "block"]
    assert verify_event_log(config.events_path, pub_pem).ok

    httpx.delete(f"{_PGW}/metrics/job/bossyk-gate/run_label/{run_label}", timeout=10.0)
