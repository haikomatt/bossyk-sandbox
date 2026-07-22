from __future__ import annotations

import threading

from bossyk_sandbox.gate import Gate, TwoSpeedGate
from bossyk_sandbox.instruments.base import InstrumentVerdict, ProposedAction, Verdict
from bossyk_sandbox.instruments.hardcoded_rule import RequireLookupBeforeCancel


class _BlockingSlowInstrument:
    """A fake `SlowInstrument` that blocks on an event until the test
    releases it — used to prove the fast decision doesn't wait on it."""

    def __init__(
        self,
        release: threading.Event,
        name: str = "slow-fake",
        wait_timeout: float = 5.0,
    ) -> None:
        self._release = release
        self._wait_timeout = wait_timeout
        self.was_called = threading.Event()
        self.name = name

    def annotate(
        self, proposed: ProposedAction, history: list[ProposedAction]
    ) -> InstrumentVerdict:
        self.was_called.set()
        self._release.wait(timeout=self._wait_timeout)
        return InstrumentVerdict(instrument=self.name, label="goal_deviation")


class _RaisingSlowInstrument:
    """A fake `SlowInstrument` that always raises -- used to prove one
    instrument's failure (Finding 8) doesn't erase the others' verdicts."""

    name = "raising-fake"

    def annotate(
        self, proposed: ProposedAction, history: list[ProposedAction]
    ) -> InstrumentVerdict:
        raise RuntimeError("boom")


class _ImmediateSlowInstrument:
    """A fake `SlowInstrument` that returns a fixed verdict without
    blocking."""

    def __init__(self, name: str, label: str = "faithful") -> None:
        self.name = name
        self._label = label

    def annotate(
        self, proposed: ProposedAction, history: list[ProposedAction]
    ) -> InstrumentVerdict:
        return InstrumentVerdict(instrument=self.name, label=self._label)


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


# --- Finding 8: slow instruments must genuinely overlap, and be isolated
# from each other's failures/hangs -----------------------------------------


def test_two_slow_instruments_run_concurrently_not_serially() -> None:
    """Today's `_annotate` loops instruments sequentially in one submitted
    future, so the second instrument's `annotate` never starts while the
    first is still blocked. Two blocking instruments, each gated on its own
    release event, must both be *called* before either release fires --
    proving they were dispatched concurrently rather than one-after-another."""
    release_a = threading.Event()
    release_b = threading.Event()
    fake_a = _BlockingSlowInstrument(release_a, name="fake-a")
    fake_b = _BlockingSlowInstrument(release_b, name="fake-b")
    gate = Gate(instruments=[RequireLookupBeforeCancel()])
    two_speed = TwoSpeedGate(gate=gate, slow_instruments=[fake_a, fake_b])

    _decision, verdicts_future = two_speed.process(
        ProposedAction("get_reservation_details", {"reservation_id": "R1"})
    )

    assert fake_a.was_called.wait(timeout=2)
    assert fake_b.was_called.wait(timeout=2)
    assert not release_a.is_set()
    assert not release_b.is_set()
    assert not verdicts_future.done()

    release_a.set()
    release_b.set()
    verdicts = verdicts_future.result(timeout=5)
    assert verdicts == [
        InstrumentVerdict(instrument="fake-a", label="goal_deviation"),
        InstrumentVerdict(instrument="fake-b", label="goal_deviation"),
    ]

    two_speed.shutdown()


def test_one_raising_instrument_does_not_erase_the_others_verdict() -> None:
    """Today, `_annotate`'s list comprehension propagates the first
    exception straight out of the submitted future, losing every verdict --
    including the ones from instruments that succeeded. Each instrument must
    be isolated: a raising instrument contributes an "error" verdict, the
    others still land in the result, in configured order."""
    gate = Gate(instruments=[RequireLookupBeforeCancel()])
    two_speed = TwoSpeedGate(
        gate=gate,
        slow_instruments=[_RaisingSlowInstrument(), _ImmediateSlowInstrument("ok-fake")],
    )

    _decision, verdicts_future = two_speed.process(
        ProposedAction("get_reservation_details", {"reservation_id": "R1"})
    )
    verdicts = verdicts_future.result(timeout=5)

    assert len(verdicts) == 2
    assert verdicts[0].instrument == "raising-fake"
    assert verdicts[0].label == "error"
    assert "boom" in verdicts[0].detail
    assert verdicts[1] == InstrumentVerdict(instrument="ok-fake", label="faithful")

    two_speed.shutdown()


def test_hung_instrument_is_isolated_by_its_own_timeout() -> None:
    """With `annotation_timeout_s` set, an instrument that exceeds it must
    contribute an "error" verdict mentioning the timeout, while the other
    instrument's real verdict is preserved. Today `annotation_timeout_s` is
    inert (unread), so the hung instrument's real (late) verdict comes back
    instead of a timeout error."""
    release_a = threading.Event()
    fake_a = _BlockingSlowInstrument(release_a, name="hung-fake", wait_timeout=1.0)
    fake_b = _ImmediateSlowInstrument("quick-fake")
    gate = Gate(instruments=[RequireLookupBeforeCancel()])
    two_speed = TwoSpeedGate(
        gate=gate,
        slow_instruments=[fake_a, fake_b],
        annotation_timeout_s=0.05,
    )

    try:
        _decision, verdicts_future = two_speed.process(
            ProposedAction("get_reservation_details", {"reservation_id": "R1"})
        )
        verdicts = verdicts_future.result(timeout=5)

        assert verdicts[0].instrument == "hung-fake"
        assert verdicts[0].label == "error"
        assert "timeout" in verdicts[0].detail.lower()
        assert verdicts[1] == InstrumentVerdict(instrument="quick-fake", label="faithful")
    finally:
        release_a.set()
        two_speed.shutdown()
