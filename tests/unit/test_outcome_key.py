from __future__ import annotations

import pytest

from bossyk_sandbox.instruments.outcome_key import BoundaryLabel, OutcomeKey, OutcomeKeyLookup

# Finding 11: OutcomeKeyLookup.__post_init__ builds its index via a dict
# comprehension over `keys` -- a duplicate (scenario_id, step_index) is
# silently last-wins today, which lets a ground-truth authoring mistake
# overwrite an earlier, correct boundary_label without any signal.


def test_duplicate_scenario_step_key_raises_value_error() -> None:
    keys = [
        OutcomeKey(scenario_id="s1", step_index=0, boundary_label=BoundaryLabel.BENIGN),
        OutcomeKey(scenario_id="s1", step_index=0, boundary_label=BoundaryLabel.POLICY_VIOLATION),
    ]

    with pytest.raises(ValueError) as exc_info:
        OutcomeKeyLookup(keys=keys)

    message = str(exc_info.value)
    assert "s1" in message
    assert "0" in message


def test_distinct_keys_still_resolve_label_for_and_is_violation() -> None:
    keys = [
        OutcomeKey(scenario_id="s1", step_index=0, boundary_label=BoundaryLabel.BENIGN),
        OutcomeKey(scenario_id="s1", step_index=1, boundary_label=BoundaryLabel.DEVIATION),
    ]

    lookup = OutcomeKeyLookup(keys=keys)

    assert lookup.label_for("s1", 0) is BoundaryLabel.BENIGN
    assert lookup.is_violation("s1", 0) is False
    assert lookup.is_violation("s1", 1) is True
