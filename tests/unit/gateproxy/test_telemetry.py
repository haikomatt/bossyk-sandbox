"""Red-phase contract for gate telemetry export (round-2 Phase 2).

Radek's stack is Grafana + Loki + Tempo behind an OpenTelemetry collector,
with a Prometheus push gateway. Gate decisions therefore go out as OTLP
spans and log records (OTLP/HTTP, JSON encoding) and as Prometheus
counters (scrape endpoint plus optional push). Telemetry is a PROJECTION
of the signed event log, emitted alongside it, never instead of it: the
signed JSONL stays the system of record and is unaffected whether the
exporters are on, off, or failing.

Phase 0 evidence (tests/fixtures/otel/): the request shapes here were
accepted verbatim by otelcol 0.160.0 (`{"partialSuccess":{}}`, HTTP 200;
a non-hex trace id is refused with HTTP 400) and by pushgateway 1.11.3
(HTTP 200; a malformed line is refused with HTTP 400). Radek's own
collector config was not available, so this codes to the OTLP spec, not
to his pipeline.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import httpx

from bossyk_sandbox.gateproxy.telemetry import (
    DecisionCounters,
    Telemetry,
    TelemetryConfig,
    otlp_logs_request,
    otlp_trace_request,
    prometheus_text,
)

_FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "otel"

_BLOCK_EVENT: dict[str, Any] = {
    "kind": "tool_call",
    "tool_name": "bash",
    "arguments": {"command": "curl http://x.example"},
    "verdict": "block",
    "policy_id": "no-network-egress",
    "reason": "no-network-egress: network egress via 'curl' is not permitted",
    "model": "upstream-model",
    "pack_sha256": "a1b2c3",
    "gate_version": "0.1.0",
    "run_label": "leg-b",
    "timestamp": "2026-09-08T12:20:00+00:00",
}

_ALLOW_EVENT: dict[str, Any] = {
    **_BLOCK_EVENT,
    "arguments": {"command": "ls"},
    "verdict": "allow",
    "policy_id": None,
    "reason": "no instrument blocked",
}

_HELD_EVENT: dict[str, Any] = {
    **_BLOCK_EVENT,
    "verdict": "hold",
    "resolution": "held_then_blocked",
    "resolved_verdict": "block",
    "policy_id": "destructive-shell",
}

_TRACE_ID = "5b8efff798038103d269b633813fc60c"
_SPAN_ID = "eee19b7ec3c1b174"


def _attrs(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {a["key"]: a["value"]["stringValue"] for a in items}


class TestOtlpTraceRequest:
    def test_matches_the_shape_the_collector_accepted(self) -> None:
        accepted = json.loads((_FIXTURES / "otlp-trace-request.json").read_text())
        built = otlp_trace_request(_BLOCK_EVENT, trace_id=_TRACE_ID, span_id=_SPAN_ID)
        resource = built["resourceSpans"][0]
        accepted_resource = accepted["resourceSpans"][0]
        assert _attrs(resource["resource"]["attributes"])["service.name"] == "bossyk-gate"
        assert resource["scopeSpans"][0]["scope"] == accepted_resource["scopeSpans"][0]["scope"]
        span = resource["scopeSpans"][0]["spans"][0]
        accepted_span = accepted_resource["scopeSpans"][0]["spans"][0]
        assert set(span) == set(accepted_span)
        assert span["name"] == "gate.decision"
        assert span["kind"] == 1
        assert span["traceId"] == _TRACE_ID and span["spanId"] == _SPAN_ID

    def test_span_carries_the_decision_attributes(self) -> None:
        span = otlp_trace_request(_BLOCK_EVENT, trace_id=_TRACE_ID, span_id=_SPAN_ID)[
            "resourceSpans"
        ][0]["scopeSpans"][0]["spans"][0]
        attrs = _attrs(span["attributes"])
        assert attrs["bossyk.verdict"] == "block"
        assert attrs["bossyk.policy_id"] == "no-network-egress"
        assert attrs["bossyk.tool_name"] == "bash"
        assert attrs["bossyk.pack_sha256"] == "a1b2c3"
        assert attrs["bossyk.run_label"] == "leg-b"
        assert attrs["bossyk.gate_version"] == "0.1.0"
        assert attrs["bossyk.kind"] == "tool_call"
        # Arguments are never exported: telemetry carries the decision,
        # the signed log carries the evidence.
        assert "bossyk.arguments" not in attrs
        assert "curl" not in json.dumps(span)

    def test_timestamps_are_unix_nanos_as_strings_from_the_event_timestamp(self) -> None:
        span = otlp_trace_request(_BLOCK_EVENT, trace_id=_TRACE_ID, span_id=_SPAN_ID)[
            "resourceSpans"
        ][0]["scopeSpans"][0]["spans"][0]
        assert span["startTimeUnixNano"] == "1788870000000000000"
        assert isinstance(span["endTimeUnixNano"], str)
        assert int(span["endTimeUnixNano"]) >= int(span["startTimeUnixNano"])

    def test_held_decision_exports_resolution(self) -> None:
        span = otlp_trace_request(_HELD_EVENT, trace_id=_TRACE_ID, span_id=_SPAN_ID)[
            "resourceSpans"
        ][0]["scopeSpans"][0]["spans"][0]
        attrs = _attrs(span["attributes"])
        assert attrs["bossyk.verdict"] == "hold"
        assert attrs["bossyk.resolution"] == "held_then_blocked"
        assert attrs["bossyk.resolved_verdict"] == "block"

    def test_absent_policy_id_is_omitted_not_stringified(self) -> None:
        span = otlp_trace_request(_ALLOW_EVENT, trace_id=_TRACE_ID, span_id=_SPAN_ID)[
            "resourceSpans"
        ][0]["scopeSpans"][0]["spans"][0]
        assert "bossyk.policy_id" not in _attrs(span["attributes"])


class TestOtlpLogsRequest:
    def test_matches_the_shape_the_collector_accepted(self) -> None:
        accepted = json.loads((_FIXTURES / "otlp-logs-request.json").read_text())
        built = otlp_logs_request(_BLOCK_EVENT, trace_id=_TRACE_ID, span_id=_SPAN_ID)
        record = built["resourceLogs"][0]["scopeLogs"][0]["logRecords"][0]
        accepted_record = accepted["resourceLogs"][0]["scopeLogs"][0]["logRecords"][0]
        assert set(record) == set(accepted_record)
        assert record["body"] == {"stringValue": _BLOCK_EVENT["reason"]}
        assert record["traceId"] == _TRACE_ID and record["spanId"] == _SPAN_ID
        assert record["timeUnixNano"] == "1788870000000000000"

    def test_severity_tracks_the_verdict(self) -> None:
        def severity(event: dict[str, Any]) -> tuple[int, str]:
            record = otlp_logs_request(event, trace_id=_TRACE_ID, span_id=_SPAN_ID)["resourceLogs"][
                0
            ]["scopeLogs"][0]["logRecords"][0]
            return record["severityNumber"], record["severityText"]

        assert severity(_ALLOW_EVENT) == (9, "INFO")
        assert severity(_BLOCK_EVENT) == (13, "WARN")
        assert severity(_HELD_EVENT) == (13, "WARN")


class TestPrometheusText:
    def test_counters_by_verdict_policy_and_resolution(self) -> None:
        counters = DecisionCounters()
        for event in (_ALLOW_EVENT, _ALLOW_EVENT, _BLOCK_EVENT, _HELD_EVENT):
            counters.observe(event)
        text = prometheus_text(counters)
        assert 'bossyk_gate_decisions_total{verdict="allow",policy_id=""} 2' in text
        assert (
            'bossyk_gate_decisions_total{verdict="block",policy_id="no-network-egress"} 1' in text
        )
        assert 'bossyk_gate_decisions_total{verdict="hold",policy_id="destructive-shell"} 1' in text
        assert 'bossyk_gate_holds_total{resolution="held_then_blocked"} 1' in text
        assert 'bossyk_gate_blocks_total{policy_id="no-network-egress"} 1' in text
        assert "# TYPE bossyk_gate_decisions_total counter" in text

    def test_exposition_lines_are_well_formed(self) -> None:
        """The grammar pushgateway 1.11.3 accepted (tests/fixtures/otel/
        pushgateway-request.txt): `name{label="v",...} <number>`, `# TYPE`
        and `# HELP` comment lines, trailing newline."""
        counters = DecisionCounters()
        counters.observe(_BLOCK_EVENT)
        text = prometheus_text(counters)
        assert text.endswith("\n")
        sample = re.compile(r'^[a-z_]+(\{([a-z_]+="[^"]*",?)*\})? \d+$')
        for line in text.splitlines():
            assert line.startswith("# ") or sample.match(line), line

    def test_label_values_are_escaped(self) -> None:
        counters = DecisionCounters()
        counters.observe({**_BLOCK_EVENT, "policy_id": 'we"ird\\pol'})
        assert 'policy_id="we\\"ird\\\\pol"' in prometheus_text(counters)

    def test_empty_counters_still_declare_types(self) -> None:
        text = prometheus_text(DecisionCounters())
        assert "# TYPE bossyk_gate_decisions_total counter" in text


class _Capture:
    def __init__(self, status: int = 200) -> None:
        self.requests: list[httpx.Request] = []
        self.status = status

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(self.status, json={"partialSuccess": {}})

    def by_path(self, suffix: str) -> list[httpx.Request]:
        return [r for r in self.requests if r.url.path.endswith(suffix)]


def _telemetry(capture: _Capture, *, otlp: bool = True, pushgateway: bool = True) -> Telemetry:
    return Telemetry(
        TelemetryConfig(
            otlp_endpoint="http://collector.invalid:4318" if otlp else None,
            pushgateway_url="http://pgw.invalid:9091" if pushgateway else None,
        ),
        transport=httpx.MockTransport(capture),
    )


class TestTelemetryExport:
    def test_one_span_and_one_log_record_per_decision(self) -> None:
        capture = _Capture()
        telemetry = _telemetry(capture, pushgateway=False)
        telemetry.record(_ALLOW_EVENT)
        telemetry.record(_BLOCK_EVENT)
        assert len(capture.by_path("/v1/traces")) == 2
        assert len(capture.by_path("/v1/logs")) == 2
        trace_ids = [
            json.loads(r.content)["resourceSpans"][0]["scopeSpans"][0]["spans"][0]["traceId"]
            for r in capture.by_path("/v1/traces")
        ]
        assert all(re.fullmatch(r"[0-9a-f]{32}", t) for t in trace_ids)
        assert len(set(trace_ids)) == 2
        assert all(r.headers["content-type"] == "application/json" for r in capture.requests)

    def test_log_record_links_to_its_span(self) -> None:
        capture = _Capture()
        _telemetry(capture, pushgateway=False).record(_BLOCK_EVENT)
        span = json.loads(capture.by_path("/v1/traces")[0].content)["resourceSpans"][0][
            "scopeSpans"
        ][0]["spans"][0]
        record = json.loads(capture.by_path("/v1/logs")[0].content)["resourceLogs"][0]["scopeLogs"][
            0
        ]["logRecords"][0]
        assert (record["traceId"], record["spanId"]) == (span["traceId"], span["spanId"])

    def test_pushgateway_put_grouped_by_job_and_run_label(self) -> None:
        capture = _Capture()
        _telemetry(capture, otlp=False).record(_BLOCK_EVENT)
        (push,) = capture.requests
        assert push.method == "PUT"
        assert push.url.path == "/metrics/job/bossyk-gate/run_label/leg-b"
        body = push.content.decode()
        assert (
            'bossyk_gate_decisions_total{verdict="block",policy_id="no-network-egress"} 1' in body
        )

    def test_counters_accumulate_across_decisions(self) -> None:
        capture = _Capture()
        telemetry = _telemetry(capture, otlp=False)
        telemetry.record(_ALLOW_EVENT)
        telemetry.record(_ALLOW_EVENT)
        assert 'verdict="allow",policy_id=""} 2' in capture.requests[-1].content.decode()
        assert 'verdict="allow",policy_id=""} 2' in telemetry.prometheus_text()

    def test_exporters_off_emit_nothing(self) -> None:
        capture = _Capture()
        telemetry = _telemetry(capture, otlp=False, pushgateway=False)
        telemetry.record(_BLOCK_EVENT)
        assert capture.requests == []
        assert telemetry.enabled is False
        # The scrape text still counts, so /gate/metrics works with no egress.
        assert 'verdict="block"' in telemetry.prometheus_text()

    def test_export_failure_is_logged_not_raised(self) -> None:
        capture = _Capture(status=503)
        _telemetry(capture).record(_BLOCK_EVENT)

    def test_transport_error_is_logged_not_raised(self) -> None:
        def explode(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("collector down")

        telemetry = Telemetry(
            TelemetryConfig(otlp_endpoint="http://c.invalid:4318", pushgateway_url=None),
            transport=httpx.MockTransport(explode),
        )
        telemetry.record(_BLOCK_EVENT)
