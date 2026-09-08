from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field

from bossyk_sandbox.instruments.base import (
    Decision,
    Instrument,
    InstrumentVerdict,
    ObservedAction,
    ProposedAction,
    SlowInstrument,
    Verdict,
    _action_of,
)
from bossyk_sandbox.instruments.drift import ERROR_LABEL


@dataclass
class Gate:
    """The interception GATE: holds each proposed tool call synchronously,
    scores it against the configured instruments (pre-execution), and
    returns allow/block. Stops at the first instrument that blocks.

    Internal history (slice 2, P5) is `ProposedAction | ObservedAction`: a
    caller that executed the tool and has its result can `record` an
    `ObservedAction` so outcome-aware fast rules (`RequirePassedCheck`) can
    see it via `Gate.score`'s internal history. The public `.history`
    property always unwraps back to bare `ProposedAction` (`_action_of`), so
    every existing caller reading `gate.history` -- tests, `live_replay.py`'s
    `LiveRunResult.executed`, the console -- stays byte-identical regardless
    of whether an entry was recorded with or without an observed result.
    """

    instruments: list[Instrument]
    _history: list[ProposedAction | ObservedAction] = field(default_factory=list)

    @property
    def history(self) -> list[ProposedAction]:
        """The recorded actions, unwrapped to bare `ProposedAction` and in
        commit order, regardless of whether some entries carry an observed
        result internally.

        Read-only: mutating the returned list does not affect the gate's
        internal history.
        """
        return [_action_of(item) for item in self._history]

    def score(self, proposed: ProposedAction) -> Decision:
        """Score a proposed action without recording it in session history.

        Used by callers (e.g. the console, the gate proxy) that need to hold
        the action open for a manual override before it's committed via
        `record`. Precedence: the first BLOCK wins outright; otherwise the
        first HOLD; otherwise ALLOW. A HOLD is a request for a decision
        outside the gate -- the caller resolves it, the gate never does.
        """
        held: Decision | None = None
        for instrument in self.instruments:
            decision = instrument.score(proposed, self._history)
            if decision.verdict is Verdict.BLOCK:
                return decision
            if decision.verdict is Verdict.HOLD and held is None:
                held = decision
        return held or Decision(Verdict.ALLOW, "no instrument blocked")

    def record(self, item: ProposedAction | ObservedAction) -> None:
        """Commit a proposed (or observed, with its tool result) action to
        session history. Passing a bare `ProposedAction` (the pre-P5
        behaviour) is still fully supported -- only a caller that has a real
        tool result to attach needs `ObservedAction`."""
        self._history.append(item)


@dataclass
class TwoSpeedGate:
    """Phase 1's two-speed split: `gate` makes the synchronous allow/block
    decision (fast path); `slow_instruments` (drift, policy) annotate the
    same proposed action concurrently -- each submitted to its own worker so
    they genuinely overlap, without the gate decision waiting on any of them.
    Annotations inform evidence + orthogonality; they never gate. Each
    instrument is isolated from the others' hangs/failures: a timeout or
    exception in one produces an "error" verdict for that instrument alone,
    the rest still land in the result, in configured order."""

    gate: Gate
    slow_instruments: list[SlowInstrument]
    # Per-instrument timeout (None = wait forever, preserving benchmark
    # behaviour where first-call model loads are slow). Read by `_gather`:
    # an instrument that exceeds it contributes an "error" verdict instead
    # of holding up the others.
    annotation_timeout_s: float | None = None
    _executor: ThreadPoolExecutor = field(init=False)

    def __post_init__(self) -> None:
        # `_gather` occupies a worker of its own while it waits on the
        # per-instrument futures, so sizing strictly to `len(slow_instruments)`
        # can starve it. Scale with the instrument count, floored at the old
        # fixed default.
        max_workers = max(4, 2 * (len(self.slow_instruments) + 1))
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="slow-instrument"
        )

    def process(self, proposed: ProposedAction) -> tuple[Decision, Future[list[InstrumentVerdict]]]:
        """Returns the fast decision immediately (already committed to
        history) plus a Future for the slow instruments' verdicts, submitted
        against the history snapshot as of just before this action.

        This path deliberately records PROPOSALS, not successful executions
        (scripted benchmark semantics — Phase 1-4 committed results depend
        on it), unlike the live agent path where history means successful
        execution.
        """
        history_snapshot = self.gate.history
        decision = self.gate.score(proposed)
        self.gate.record(proposed)
        instrument_futures = [
            self._executor.submit(instrument.annotate, proposed, history_snapshot)
            for instrument in self.slow_instruments
        ]
        return decision, self._executor.submit(self._gather, instrument_futures)

    def _gather(self, futures: list[Future[InstrumentVerdict]]) -> list[InstrumentVerdict]:
        """Collects each instrument's future in configured order, isolating
        the caller from any single instrument's timeout or exception."""
        verdicts: list[InstrumentVerdict] = []
        for instrument, future in zip(self.slow_instruments, futures, strict=True):
            try:
                verdicts.append(future.result(timeout=self.annotation_timeout_s))
            except TimeoutError:
                verdicts.append(
                    InstrumentVerdict(
                        instrument=instrument.name,
                        label=ERROR_LABEL,
                        detail=f"annotation timeout ({self.annotation_timeout_s}s) exceeded",
                    )
                )
            except Exception as exc:
                verdicts.append(
                    InstrumentVerdict(
                        instrument=instrument.name, label=ERROR_LABEL, detail=str(exc)
                    )
                )
        return verdicts

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True)
