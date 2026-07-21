from __future__ import annotations

from dataclasses import dataclass, field

from bossyk_sandbox.instruments.base import Decision, Instrument, ProposedAction, Verdict


@dataclass
class Gate:
    """The interception GATE: holds each proposed tool call synchronously,
    scores it against the configured instruments (pre-execution), and
    returns allow/block. Stops at the first instrument that blocks.
    """

    instruments: list[Instrument]
    _history: list[ProposedAction] = field(default_factory=list)

    def score(self, proposed: ProposedAction) -> Decision:
        """Score a proposed action without recording it in session history.

        Used by callers (e.g. the console) that need to hold the action open
        for a manual override before it's committed via `record`.
        """
        for instrument in self.instruments:
            decision = instrument.score(proposed, self._history)
            if decision.verdict is Verdict.BLOCK:
                return decision
        return Decision(Verdict.ALLOW, "no instrument blocked")

    def record(self, proposed: ProposedAction) -> None:
        """Commit a proposed action to session history."""
        self._history.append(proposed)

    def evaluate(self, proposed: ProposedAction) -> Decision:
        """Score and immediately commit a proposed action. The synchronous
        hold -> score -> allow/block path used when no manual override step
        is needed."""
        decision = self.score(proposed)
        self.record(proposed)
        return decision
