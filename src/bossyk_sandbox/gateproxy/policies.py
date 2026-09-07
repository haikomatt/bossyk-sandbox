"""RED-phase skeleton -- see tests/unit/gateproxy/test_policies.py."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_policy_pack(path: Path) -> Any:
    raise NotImplementedError("gateproxy Green phase pending")


def compile_instruments(pack: Any, *, workspace_root: Path) -> Any:
    raise NotImplementedError("gateproxy Green phase pending")
