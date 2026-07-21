from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol


class Verdict(StrEnum):
    ALLOW = "allow"
    BLOCK = "block"


@dataclass(frozen=True)
class ProposedAction:
    tool_name: str
    arguments: dict[str, Any]
    declared_intent: str | None = None


@dataclass(frozen=True)
class Decision:
    verdict: Verdict
    reason: str


@dataclass(frozen=True)
class InstrumentVerdict:
    """A slow instrument's per-step annotation. Not a gate decision — these
    inform evidence and orthogonality, they never block (Phase 1 two-speed
    split)."""

    instrument: str
    label: str
    detail: str = ""


class Instrument(Protocol):
    """Scores a proposed action against the session history so far, fast
    enough to gate synchronously."""

    def score(self, proposed: ProposedAction, history: list[ProposedAction]) -> Decision: ...


class SlowInstrument(Protocol):
    """Annotates a proposed action with a verdict too slow to gate on
    (Phase 1 drift + policy judges). Runs concurrently with the fast path."""

    name: str

    def annotate(
        self, proposed: ProposedAction, history: list[ProposedAction]
    ) -> InstrumentVerdict: ...
