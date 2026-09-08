"""HOLD approver hooks (round-2 Phase 1). RED-phase skeleton: importable
names, no behaviour -- the implementation lands in Green."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass(frozen=True)
class HoldRequest:
    tool_name: str
    arguments: dict[str, Any]
    policy_id: str | None
    reason: str
    run_label: str


Approver = Callable[[HoldRequest], bool | None]


def command_approver(command: list[str], *, timeout_s: float) -> Approver:
    raise NotImplementedError


def url_approver(
    url: str, *, timeout_s: float, transport: httpx.BaseTransport | None = None
) -> Approver:
    raise NotImplementedError
