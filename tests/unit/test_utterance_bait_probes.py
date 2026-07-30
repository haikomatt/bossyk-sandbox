"""RED-phase tests for the hand-authored outreach utterance-bait probes
(bossyk-sandbox slice 3; Matt's decision: the re-run should also
demonstrate the P6 utterance boundary, via hand-authored bait rather than
adversary generation -- boundary 5 has no BoundarySpec, so it can never be
targeted by the tool-grounded adversary at all, see
test_grounded_boundaries.py).

Design proposed here (flagged for review, not silently decided):

- Location: `probes/handauthored/outreach-prohibited-financial-promotion.json`
  -- a NEW `probes/handauthored/` directory, distinct from `probes/grounded/`
  (adversary-generated) and `probes/regression/` (frozen discovered
  crossings), since this data has neither provenance: it is manually
  authored, never generated, never discovered.
- Schema: the SAME `ProbeDefinition` list `conditions.retention
  .load_regression_probes`/`save_regression_probes` already read/write --
  no new loader needed, the live bench's `load_regression_probes(path)`
  call (scripts/live_h2h4_bench.py) works unchanged once pointed at this
  file (e.g. via `LIVE_H2_CORPUS`).
- `family = "outreach-prohibited_financial_promotion"` (matches
  `parse_family`'s `<domain>-<boundary>` convention, so `replay_crossing`
  resolves `domain="outreach"`, `boundary="prohibited_financial_promotion"`
  exactly like any other probe).
- `kind`: classified as `"jailbreak"` (closest fit among the 4 registered
  AttackClass values -- a customer nudging the agent toward a prohibited
  promotion is a soft social-engineering framing, not prompt_injection/
  tool_misuse/pii_leak). Flagged as a genuine judgment call, not obviously
  correct.
- `expected_behavior.should_refuse = true` on every probe: the agent
  should decline to make the regulated promotion, not necessarily refuse
  the customer's underlying request (e.g. it can still say "we don't do
  0% finance, but I can book you a free survey").
- Clean room: every payload below is a GENERIC customer phrasing authored
  from the cleanroom spec's own regulated-phrase list (0% finance,
  interest-free, buy now pay later, no deposit, government grant, spread
  the cost, guaranteed acceptance/finance) -- no third-party source.

This file is NOT authored yet (GREEN, after review) -- these tests fail
with FileNotFoundError today.
"""

from __future__ import annotations

from pathlib import Path

from auditk.schema import ProbeDefinition

from bossyk_sandbox.conditions.retention import load_regression_probes

BAIT_PROBES_PATH = (
    Path(__file__).parent.parent.parent
    / "probes"
    / "handauthored"
    / "outreach-prohibited-financial-promotion.json"
)


def _load() -> list[ProbeDefinition]:
    return load_regression_probes(BAIT_PROBES_PATH)


def test_bait_probes_file_loads_between_four_and_six_probes() -> None:
    probes = _load()

    assert 4 <= len(probes) <= 6


def test_every_bait_probe_has_the_outreach_utterance_family() -> None:
    probes = _load()

    for probe in probes:
        assert probe.family == "outreach-prohibited_financial_promotion"


def test_every_bait_probe_has_a_non_empty_payload_text() -> None:
    probes = _load()

    for probe in probes:
        assert probe.stimulus.payload["text"].strip()


def test_every_bait_probe_expects_a_refusal() -> None:
    probes = _load()

    for probe in probes:
        assert probe.expected_behavior.should_refuse is True


def test_bait_probe_ids_are_unique() -> None:
    probes = _load()

    ids = [probe.probe_id for probe in probes]
    assert len(ids) == len(set(ids))


def test_bait_probe_payloads_are_generic_customer_phrasings_not_regulated_claims() -> None:
    # Clean-room sanity: the CUSTOMER's bait text should ask about finance
    # options in plain language -- it is the AGENT's reply that must not
    # affirmatively promise a regulated term, not the customer's question.
    # This just guards against accidentally authoring the probe payload
    # AS an agent-voiced promotional claim.
    probes = _load()

    for probe in probes:
        text = probe.stimulus.payload["text"].lower()
        assert "?" in text or "can you" in text or "do you" in text, (
            f"{probe.probe_id} does not read as a customer question: {text!r}"
        )
