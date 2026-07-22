from __future__ import annotations

import os
from pathlib import Path

import pytest

from bossyk_sandbox.evidence.pack import build_and_sign, verify_offline
from bossyk_sandbox.instruments.base import InstrumentVerdict, ProposedAction
from bossyk_sandbox.instruments.outcome_key import OutcomeKeyLookup
from bossyk_sandbox.scenarios.loader import load_scenarios, outcome_keys
from bossyk_sandbox.scenarios.runner import (
    VERDICT_METADATA_KEY,
    run_scenario,
    to_gate_outcomes,
    to_membership,
)
from bossyk_sandbox.scoring.confusion import b2_safety_weighted, b3_bind_headline, binary_confusion
from bossyk_sandbox.scoring.orthogonality import orthogonality_table

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_SANDBOX_E2E") != "1",
    reason="Set RUN_SANDBOX_E2E=1 to run the multi-instrument E2E test.",
)


class _StubDriftInstrument:
    """Deterministic stand-in for the real llm-judge@0.3 drift instrument:
    flags drift whenever declared_intent doesn't literally match the tool
    being called (a crude but deterministic proxy for the scripted
    scenarios' scope-deviation steps)."""

    name = "drift"

    def annotate(
        self, proposed: ProposedAction, history: list[ProposedAction]
    ) -> InstrumentVerdict:
        intent = (proposed.declared_intent or "").lower()
        if proposed.tool_name == "cancel_reservation" and "cancel" not in intent:
            return InstrumentVerdict(
                instrument=self.name, label="goal_deviation", detail="scripted stub"
            )
        if proposed.tool_name == "update_reservation_flights" and "rebook" not in intent:
            return InstrumentVerdict(
                instrument=self.name, label="goal_deviation", detail="scripted stub"
            )
        return InstrumentVerdict(instrument=self.name, label="faithful", detail="scripted stub")


class _StubPolicyInstrument:
    """Deterministic stand-in for the real PolicyAwareJudge: flags a
    violation whenever the fast-path gate itself would also block (the
    scripted scenarios' unauthorised/wrong-reservation steps)."""

    name = "policy"

    def __init__(self) -> None:
        self._seen_lookups: set[tuple[str, str]] = set()

    def annotate(
        self, proposed: ProposedAction, history: list[ProposedAction]
    ) -> InstrumentVerdict:
        if proposed.tool_name == "get_reservation_details":
            return InstrumentVerdict(instrument=self.name, label="faithful", detail="scripted stub")

        reservation_id = proposed.arguments.get("reservation_id")
        looked_up = any(
            call.tool_name == "get_reservation_details"
            and call.arguments.get("reservation_id") == reservation_id
            for call in history
        )
        if looked_up:
            return InstrumentVerdict(instrument=self.name, label="faithful", detail="scripted stub")
        return InstrumentVerdict(
            instrument=self.name, label="instruction_noncompliance", detail="scripted stub"
        )


def test_multi_instrument_scenario_set_end_to_end(tmp_path: Path) -> None:
    scenarios = load_scenarios()
    lookup = OutcomeKeyLookup(keys=outcome_keys(scenarios))

    all_memberships = []
    all_gate_outcomes = []
    for scenario in scenarios:
        trace, scored_steps = run_scenario(
            scenario, slow_instruments=[_StubDriftInstrument(), _StubPolicyInstrument()]
        )

        # (a) verdicts from all three instruments land in the signed pack:
        # drift + policy directly on each step, outcome via the joined key.
        for scored in scored_steps:
            verdict_metadata = scored.step.metadata[VERDICT_METADATA_KEY]
            assert verdict_metadata["drift"]["label"] is not None
            assert verdict_metadata["policy"]["label"] is not None
            assert lookup.label_for(scenario.scenario_id, scored.step_index) is not None

        from auditk.attestation.signer import generate_keypair

        priv_path, pub_path = generate_keypair(tmp_path / f"{scenario.scenario_id}-key")
        pack = build_and_sign(trace, agent_version="0.1.0", signer_key_path=priv_path)
        assert verify_offline(pack, pub_path.read_text()) is True

        all_memberships.extend(to_membership(scored_steps, lookup))
        all_gate_outcomes.extend(to_gate_outcomes(scored_steps, lookup))

    # (b) the tables compute over the full scenario set
    table = orthogonality_table(all_memberships)
    assert sum(row.count for row in table) == len(all_memberships)
    assert len(all_memberships) > 0

    matrix = binary_confusion(all_gate_outcomes)
    b2_safety_weighted(matrix)
    b3_bind_headline(matrix)
    assert matrix.true_bind + matrix.false_bind + matrix.true_no_bind + matrix.false_no_bind == len(
        all_gate_outcomes
    )
