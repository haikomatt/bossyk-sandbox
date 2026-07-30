from __future__ import annotations

from collections.abc import Sequence
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
class ObservedAction:
    """A prior action paired with the RESULT its tool call actually
    returned (bossyk-sandbox slice 2, P5; scope-doc D3) -- mirrors the
    existing `standing.TimedAction` precedent, which already wraps a
    `ProposedAction` with extra data (there: `at`/`amount`; here: `result`).

    `ProposedAction` itself is deliberately NOT extended with a `result`
    field: a proposal has no result, only an observed action does. This is
    what makes "the agent called the check but ignored its RESULT"
    governable (`RequirePassedCheck`), which precedence-only gating
    (`RequireLookupBeforeCancel`) cannot express."""

    action: ProposedAction
    result: Any


def _action_of(item: ProposedAction | ObservedAction) -> ProposedAction:
    """Unwraps a history entry to its underlying `ProposedAction`, exactly
    as `standing._action_of` already does for `TimedAction`. Every existing
    instrument that only reads `.tool_name`/`.arguments` off history items
    must go through this unwrapper so it keeps behaving identically whether
    a given entry is a bare `ProposedAction` or an `ObservedAction`."""
    return item.action if isinstance(item, ObservedAction) else item


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
    enough to gate synchronously.

    `history` is `Sequence[ProposedAction | ObservedAction]` (widened for
    slice 2, P5): a fast rule that needs a prior check's RESULT (not merely
    that it was called) reads it off an `ObservedAction` entry via
    `_action_of`/`isinstance`. Existing precedence-only rules that never
    cared about results keep working unchanged as long as they unwrap
    entries with `_action_of` before reading `.tool_name`/`.arguments`."""

    def score(
        self, proposed: ProposedAction, history: Sequence[ProposedAction | ObservedAction]
    ) -> Decision: ...


class SlowInstrument(Protocol):
    """Annotates a proposed action with a verdict too slow to gate on
    (Phase 1 drift + policy judges). Runs concurrently with the fast path."""

    name: str

    def annotate(
        self, proposed: ProposedAction, history: list[ProposedAction]
    ) -> InstrumentVerdict: ...
