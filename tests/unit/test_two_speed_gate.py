from __future__ import annotations

import threading

from bossyk_sandbox.gate import Gate, TwoSpeedGate
from bossyk_sandbox.instruments.base import InstrumentVerdict, ProposedAction, Verdict
from bossyk_sandbox.instruments.hardcoded_rule import RequireLookupBeforeCancel


class _BlockingSlowInstrument:
    """A fake `SlowInstrument` that blocks on an event until the test
    releases it — used to prove the fast decision doesn't wait on it."""

    name = "slow-fake"

    def __init__(self, release: threading.Event) -> None:
        self._release = release
        self.was_called = threading.Event()

    def annotate(
        self, proposed: ProposedAction, history: list[ProposedAction]
    ) -> InstrumentVerdict:
        self.was_called.set()
        self._release.wait(timeout=5)
        return InstrumentVerdict(instrument=self.name, label="goal_deviation")


def test_fast_decision_returns_before_slow_instrument_completes() -> None:
    release = threading.Event()
    slow = _BlockingSlowInstrument(release)
    gate = Gate(instruments=[RequireLookupBeforeCancel()])
    two_speed = TwoSpeedGate(gate=gate, slow_instruments=[slow])

    decision, verdicts_future = two_speed.process(
        ProposedAction("get_reservation_details", {"reservation_id": "R1"})
    )

    # The fast decision is already back...
    assert decision.verdict is Verdict.ALLOW
    # ...while the slow instrument is still blocked mid-flight.
    assert slow.was_called.wait(timeout=2)
    assert not verdicts_future.done()

    release.set()
    verdicts = verdicts_future.result(timeout=5)
    assert verdicts == [InstrumentVerdict(instrument="slow-fake", label="goal_deviation")]

    two_speed.shutdown()


def test_slow_instruments_see_history_as_of_before_the_proposed_action() -> None:
    seen_history: list[list[ProposedAction]] = []

    class _RecordingInstrument:
        name = "recorder"

        def annotate(
            self, proposed: ProposedAction, history: list[ProposedAction]
        ) -> InstrumentVerdict:
            seen_history.append(list(history))
            return InstrumentVerdict(instrument=self.name, label="faithful")

    gate = Gate(instruments=[RequireLookupBeforeCancel()])
    two_speed = TwoSpeedGate(gate=gate, slow_instruments=[_RecordingInstrument()])

    two_speed.process(ProposedAction("get_reservation_details", {"reservation_id": "R1"}))[
        1
    ].result()
    two_speed.process(ProposedAction("cancel_reservation", {"reservation_id": "R1"}))[1].result()

    assert seen_history[0] == []
    assert seen_history[1] == [ProposedAction("get_reservation_details", {"reservation_id": "R1"})]

    two_speed.shutdown()
