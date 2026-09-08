"""HOLD approver hooks (round-2 Phase 1: human-in-the-loop).

A HOLD verdict pauses a proposed action for a decision outside the gate.
In a ``pi --print`` batch run there is no human at a console, so the
decision comes from an external approver: a command (the held action as
JSON on stdin; exit 0 approves, any other exit denies) or a URL (JSON
POST; the response body's ``decision`` is ``allow`` or ``block``). Each
hook has a timeout. A timeout, a missing command, an HTTP error or an
unrecognised answer all yield ``None`` -- "no decision" -- and the proxy
resolves the hold by the policy's own ``on_hold`` default. The approver is
only ever shown what the signed event already records about the action.

This is the mechanism behind an AI Act Art. 14 human-oversight
line: the gate can stop and ask; it does not claim a human always answers.
"""

from __future__ import annotations

import json
import logging
import subprocess
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import httpx

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HoldRequest:
    """What the approver sees: the held action and why it was held."""

    tool_name: str
    arguments: dict[str, Any]
    policy_id: str | None
    reason: str
    run_label: str


# True approves, False denies, None means no decision was reached (timeout
# or failure) and the policy's `on_hold` default applies.
Approver = Callable[[HoldRequest], bool | None]


def command_approver(command: list[str], *, timeout_s: float) -> Approver:
    """An approver that runs ``command`` with the request as JSON on stdin."""

    def approve(request: HoldRequest) -> bool | None:
        try:
            completed = subprocess.run(
                command,
                input=json.dumps(asdict(request)),
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired:
            logger.warning("hold_approver_timeout", extra={"command": command[0]})
            return None
        except OSError as exc:
            logger.warning("hold_approver_unrunnable", extra={"command": command[0], "err": exc})
            return None
        return completed.returncode == 0

    return approve


def url_approver(
    url: str, *, timeout_s: float, transport: httpx.BaseTransport | None = None
) -> Approver:
    """An approver that POSTs the request as JSON and reads ``decision``
    from the JSON reply. ``transport`` is injectable for tests only."""

    def approve(request: HoldRequest) -> bool | None:
        try:
            with httpx.Client(timeout=timeout_s, transport=transport) as client:
                response = client.post(url, json=asdict(request))
                response.raise_for_status()
                decision = response.json().get("decision")
        except (httpx.HTTPError, ValueError, AttributeError) as exc:
            logger.warning("hold_approver_failed", extra={"url": url, "err": str(exc)})
            return None
        if decision == "allow":
            return True
        if decision == "block":
            return False
        logger.warning("hold_approver_unrecognised_decision", extra={"decision": decision})
        return None

    return approve
