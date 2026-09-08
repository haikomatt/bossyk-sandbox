"""Gate telemetry: OTLP spans + log records and Prometheus counters.

A projection of the signed event log for an operator's existing
observability stack (typically Grafana + Loki + Tempo behind an OTel
collector, plus a Prometheus push gateway). Emitted alongside the signed
JSONL, never instead of it: the signed log is the system of record and is
written before any of this runs; an exporter that is off, slow, or
failing changes nothing about what the gate decided or recorded.

Wire formats are built by hand as plain dicts / text -- the same
no-SDK idiom as auditk's generic OTel adapter -- and were validated
against real receivers (otelcol 0.160.0, pushgateway 1.11.3; see
tests/fixtures/otel/). OTLP/HTTP JSON encoding: ids are lowercase hex
(32/16 chars), nanosecond timestamps are decimal strings, attributes are
``{key, value: {stringValue}}``. Arguments are never exported; telemetry
carries the decision, the signed log carries the evidence.
"""

from __future__ import annotations

import logging
import secrets
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import httpx

from bossyk_sandbox.gateproxy import __version__ as _GATE_VERSION

logger = logging.getLogger(__name__)

_SCOPE = {"name": "bossyk_sandbox.gateproxy", "version": _GATE_VERSION}
_JOB = "bossyk-gate"

# Event fields exported as span/log attributes, in this order. Absent or
# null fields are omitted rather than stringified.
_EXPORTED_FIELDS = (
    "verdict",
    "resolution",
    "resolved_verdict",
    "policy_id",
    "tool_name",
    "kind",
    "pack_sha256",
    "run_label",
    "gate_version",
    "model",
    # Stamped by per-identity policy packs (round 2); exported here
    # so cross-tenant scoping is enforced in the exporter, not by
    # convention. Omitted when the gate is not identity-aware.
    "caller_identity",
    "identity_source",
)


@dataclass(frozen=True)
class TelemetryConfig:
    otlp_endpoint: str | None = None
    pushgateway_url: str | None = None
    service_name: str = "bossyk-gate"


def _string_attrs(pairs: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"key": k, "value": {"stringValue": str(v)}} for k, v in pairs.items() if v is not None]


def _event_attrs(event: dict[str, Any]) -> list[dict[str, Any]]:
    return _string_attrs({f"bossyk.{k}": event.get(k) for k in _EXPORTED_FIELDS})


def _unix_nanos(timestamp: Any) -> int:
    """The event's ISO timestamp as Unix nanoseconds; now() if unparseable."""
    try:
        parsed = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return time.time_ns()
    return int(parsed.timestamp() * 1_000_000_000)


def _resource(service_name: str) -> dict[str, Any]:
    return {
        "attributes": _string_attrs(
            {"service.name": service_name, "service.version": _GATE_VERSION}
        )
    }


def otlp_trace_request(
    event: dict[str, Any], *, trace_id: str, span_id: str, service_name: str = _JOB
) -> dict[str, Any]:
    """One `gate.decision` span (kind INTERNAL) for one event, as an
    OTLP/HTTP JSON ExportTraceServiceRequest."""
    start = _unix_nanos(event.get("timestamp"))
    return {
        "resourceSpans": [
            {
                "resource": _resource(service_name),
                "scopeSpans": [
                    {
                        "scope": _SCOPE,
                        "spans": [
                            {
                                "traceId": trace_id,
                                "spanId": span_id,
                                "name": "gate.decision",
                                "kind": 1,
                                "startTimeUnixNano": str(start),
                                "endTimeUnixNano": str(max(start, time.time_ns())),
                                "attributes": _event_attrs(event),
                                "status": {"code": 0},
                            }
                        ],
                    }
                ],
            }
        ]
    }


def _severity(event: dict[str, Any]) -> tuple[int, str]:
    return (9, "INFO") if event.get("verdict") == "allow" else (13, "WARN")


def otlp_logs_request(
    event: dict[str, Any], *, trace_id: str, span_id: str, service_name: str = _JOB
) -> dict[str, Any]:
    """One log record per event (body = the decision reason), linked to
    the decision span, as an OTLP/HTTP JSON ExportLogsServiceRequest."""
    number, text = _severity(event)
    return {
        "resourceLogs": [
            {
                "resource": _resource(service_name),
                "scopeLogs": [
                    {
                        "scope": _SCOPE,
                        "logRecords": [
                            {
                                "timeUnixNano": str(_unix_nanos(event.get("timestamp"))),
                                "severityNumber": number,
                                "severityText": text,
                                "body": {"stringValue": str(event.get("reason", ""))},
                                "attributes": _event_attrs(event),
                                "traceId": trace_id,
                                "spanId": span_id,
                            }
                        ],
                    }
                ],
            }
        ]
    }


class DecisionCounters:
    """In-memory Prometheus counters over gate decisions."""

    def __init__(self) -> None:
        self.decisions: Counter[tuple[str, str]] = Counter()  # (verdict, policy_id)
        self.blocks: Counter[str] = Counter()  # policy_id
        self.holds: Counter[str] = Counter()  # resolution

    def observe(self, event: dict[str, Any]) -> None:
        verdict = str(event.get("verdict", ""))
        policy_id = str(event.get("policy_id") or "")
        self.decisions[(verdict, policy_id)] += 1
        if verdict == "block":
            self.blocks[policy_id] += 1
        if verdict == "hold":
            self.holds[str(event.get("resolution") or "")] += 1


def _label(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def prometheus_text(counters: DecisionCounters) -> str:
    """Prometheus text exposition (the grammar pushgateway accepted)."""
    lines = [
        "# HELP bossyk_gate_decisions_total Gate decisions by verdict and policy.",
        "# TYPE bossyk_gate_decisions_total counter",
    ]
    for (verdict, policy_id), n in sorted(counters.decisions.items()):
        lines.append(
            f'bossyk_gate_decisions_total{{verdict="{_label(verdict)}",'
            f'policy_id="{_label(policy_id)}"}} {n}'
        )
    lines += [
        "# HELP bossyk_gate_blocks_total Gate blocks by policy.",
        "# TYPE bossyk_gate_blocks_total counter",
    ]
    for policy_id, n in sorted(counters.blocks.items()):
        lines.append(f'bossyk_gate_blocks_total{{policy_id="{_label(policy_id)}"}} {n}')
    lines += [
        "# HELP bossyk_gate_holds_total Gate holds by resolution.",
        "# TYPE bossyk_gate_holds_total counter",
    ]
    for resolution, n in sorted(counters.holds.items()):
        lines.append(f'bossyk_gate_holds_total{{resolution="{_label(resolution)}"}} {n}')
    return "\n".join(lines) + "\n"


class Telemetry:
    """Records each signed event as telemetry. `transport` is injectable
    for tests; every export failure is logged and swallowed by design --
    a telemetry outage must never become a gate outage."""

    def __init__(
        self, config: TelemetryConfig, *, transport: httpx.BaseTransport | None = None
    ) -> None:
        self.config = config
        self.counters = DecisionCounters()
        self.enabled = bool(config.otlp_endpoint or config.pushgateway_url)
        self._client = httpx.Client(timeout=5.0, transport=transport)

    def record(self, event: dict[str, Any]) -> None:
        self.counters.observe(event)
        if self.config.otlp_endpoint:
            trace_id, span_id = secrets.token_hex(16), secrets.token_hex(8)
            base = self.config.otlp_endpoint.rstrip("/")
            self._post(
                f"{base}/v1/traces", otlp_trace_request(event, **self._ids(trace_id, span_id))
            )
            self._post(f"{base}/v1/logs", otlp_logs_request(event, **self._ids(trace_id, span_id)))
        if self.config.pushgateway_url:
            run_label = str(event.get("run_label", ""))
            base = self.config.pushgateway_url.rstrip("/")
            url = f"{base}/metrics/job/{_JOB}/run_label/{run_label}"
            self._send("PUT", url, content=self.prometheus_text())

    def prometheus_text(self) -> str:
        return prometheus_text(self.counters)

    def _ids(self, trace_id: str, span_id: str) -> dict[str, str]:
        return {"trace_id": trace_id, "span_id": span_id, "service_name": self.config.service_name}

    def _post(self, url: str, body: dict[str, Any]) -> None:
        self._send("POST", url, json=body)

    def _send(self, method: str, url: str, **kwargs: Any) -> None:
        try:
            response = self._client.request(method, url, **kwargs)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning("telemetry_export_failed", extra={"url": url, "err": str(exc)})
