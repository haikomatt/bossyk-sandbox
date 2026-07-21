from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field

from bossyk_sandbox.instruments.base import (
    Decision,
    Instrument,
    InstrumentVerdict,
    ProposedAction,
    SlowInstrument,
    Verdict,
)


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


@dataclass
class TwoSpeedGate:
    """Phase 1's two-speed split: `gate` makes the synchronous allow/block
    decision (fast path); `slow_instruments` (drift, policy) annotate the
    same proposed action concurrently, without the gate decision waiting on
    them. Annotations inform evidence + orthogonality; they never gate."""

    gate: Gate
    slow_instruments: list[SlowInstrument]
    _executor: ThreadPoolExecutor = field(
        default_factory=lambda: ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="slow-instrument"
        )
    )

    def process(self, proposed: ProposedAction) -> tuple[Decision, Future[list[InstrumentVerdict]]]:
        """Returns the fast decision immediately (already committed to
        history) plus a Future for the slow instruments' verdicts, submitted
        against the history snapshot as of just before this action."""
        history_snapshot = list(self.gate._history)
        decision = self.gate.evaluate(proposed)
        verdicts_future = self._executor.submit(self._annotate, proposed, history_snapshot)
        return decision, verdicts_future

    def _annotate(
        self, proposed: ProposedAction, history: list[ProposedAction]
    ) -> list[InstrumentVerdict]:
        return [instrument.annotate(proposed, history) for instrument in self.slow_instruments]

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True)
