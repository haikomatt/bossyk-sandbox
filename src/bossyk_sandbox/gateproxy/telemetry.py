"""Gate telemetry export (round-2 Phase 2). RED-phase skeleton: importable
names, no behaviour -- the implementation lands in Green."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx


@dataclass(frozen=True)
class TelemetryConfig:
    otlp_endpoint: str | None = None
    pushgateway_url: str | None = None
    service_name: str = "bossyk-gate"


class DecisionCounters:
    def observe(self, event: dict[str, Any]) -> None:
        raise NotImplementedError


def otlp_trace_request(event: dict[str, Any], *, trace_id: str, span_id: str) -> dict[str, Any]:
    raise NotImplementedError


def otlp_logs_request(event: dict[str, Any], *, trace_id: str, span_id: str) -> dict[str, Any]:
    raise NotImplementedError


def prometheus_text(counters: DecisionCounters) -> str:
    raise NotImplementedError


class Telemetry:
    def __init__(
        self, config: TelemetryConfig, *, transport: httpx.BaseTransport | None = None
    ) -> None:
        self.config = config
        self.enabled = False

    def record(self, event: dict[str, Any]) -> None:
        raise NotImplementedError

    def prometheus_text(self) -> str:
        raise NotImplementedError
