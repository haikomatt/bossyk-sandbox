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


@dataclass(frozen=True)
class Decision:
    verdict: Verdict
    reason: str


class Instrument(Protocol):
    """Scores a proposed action against the session history so far.

    Phase 1 drops PolicyAwareJudge in behind this same interface.
    """

    def score(self, proposed: ProposedAction, history: list[ProposedAction]) -> Decision: ...
