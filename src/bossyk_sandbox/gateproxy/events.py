"""RED-phase skeleton -- see tests/unit/gateproxy/test_events.py."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class EventLog:
    path: Path
    signer_key_path: Path
    run_label: str

    def append(self, event: dict[str, Any]) -> None:
        raise NotImplementedError("gateproxy Green phase pending")


def verify_event_log(path: Path, public_key_pem: str) -> Any:
    raise NotImplementedError("gateproxy Green phase pending")
