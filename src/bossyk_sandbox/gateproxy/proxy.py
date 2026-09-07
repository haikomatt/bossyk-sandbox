"""RED-phase skeleton -- see tests/unit/gateproxy/test_proxy.py."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class GateProxyConfig:
    upstream_base_url: str
    policy_pack_path: Path
    workspace_root: Path
    signer_key_path: Path
    events_path: Path
    run_label: str
    listen: str | None = None
    uds: str | None = None


def create_app(config: GateProxyConfig, upstream: Any = None) -> Any:
    raise NotImplementedError("gateproxy Green phase pending")


def build_config(argv: list[str]) -> GateProxyConfig:
    raise NotImplementedError("gateproxy Green phase pending")
